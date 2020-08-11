#include "EventData/TrackParameters.hpp"
#include "Geometry/GeometryContext.hpp"
#include "MagneticField/MagneticFieldContext.hpp"
#include "Plugins/BFieldOptions.hpp"
#include "Plugins/BFieldUtils.hpp"
#include "Propagator/EigenStepper.hpp"
#include "Propagator/Propagator.hpp"
#include "Utilities/ParameterDefinitions.hpp"
#include "Utilities/Units.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

static void show_usage(std::string name) {
  std::cerr << "Usage: <option(s)> VALUES"
            << "Options:\n"
            << "\t-h,--help\t\tShow this help message\n"
            << "\t-t,--tracks \tSpecify the number of tracks\n"
            << "\t-p,--pt \tSpecify the pt of particle\n"
            << "\t-o,--output \tIndicator for writing propagation results\n"
            << "\t-d,--device \tSpecify the device: 'gpu' or 'cpu'\n"
            << "\t-b,--bf-map \tSpecify the path of *.txt for interpolated "
               "BField map\n"
            << std::endl;
}

using namespace Acts;

// Struct for B field
struct ConstantBField {
  ACTS_DEVICE_FUNC static Vector3D getField(const Vector3D & /*field*/) {
    return Vector3D(0., 0., 2.0*Acts::units::_T);
  }
};

// Test actor
struct VoidActor {
  struct this_result {
    bool status = false;
  };
  using result_type = this_result;

  template <typename propagator_state_t, typename stepper_t>
  void operator()(propagator_state_t &state, const stepper_t &stepper,
                  result_type &result) const {
    if(state.navigation.currentSurface!=nullptr) {
	    std::cout<<"On surface: "<<state.navigation.nextSurfaceIter<<std::endl;
     std::cout<<" state.stepping.pos = \n "<< state.stepping.pos << "state.stepping.dir = \n"<< state.stepping.dir<<std::endl; 
    } 
    return;
  }
};

// Test aborter
struct VoidAborter {

  template <typename propagator_state_t, typename stepper_t, typename result_t>
  bool operator()(propagator_state_t &state, const stepper_t &stepper,
                  result_t &result) const {
	  return false;
  }
};

//using Stepper = EigenStepper<ConstantBField>;
using Stepper = EigenStepper<InterpolatedBFieldMap3D>;
using PropagatorType = Propagator<Stepper>;
using PropResultType = PropagatorResult<typename VoidActor::result_type>;
using PropOptionsType = PropagatorOptions<VoidActor, VoidAborter>;

int main(int argc, char *argv[]) {
  if (argc < 5) {
    show_usage(argv[0]);
    return 1;
  }
  unsigned int nTracks;
  bool output = false;
  std::string device;
  std::string bFieldFileName;
  double p;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "-h") or (arg == "--help")) {
      show_usage(argv[0]);
      return 0;
    } else if (i + 1 < argc) {
      if ((arg == "-t") or (arg == "--tracks")) {
        nTracks = atoi(argv[++i]);
      } else if ((arg == "-p") or (arg == "--pt")) {
        p = atof(argv[++i])*Acts::units::_GeV;
      } else if ((arg == "-o") or (arg == "--output")) {
        output = (atoi(argv[++i]) == 1);
      } else if ((arg == "-d") or (arg == "--device")) {
        device = argv[++i];
      } else if ((arg == "-b") or (arg == "--bf-map")) {
        bFieldFileName = argv[++i];
      } else {
        std::cerr << "Unknown argument." << std::endl;
        return 1;
      }
    }
  }

  std::cout << "----- Propgation test of " << nTracks << " tracks on " << device
            << ". Writing results to obj file? " << output << " ----- "
            << std::endl;

  // Create the geometry
  // Set translation vectors
  std::vector<Acts::Vector3D> translations;
  for(unsigned int isur = 0; isur< s_surfacesSize; isur++){
    translations.push_back({(isur * 30. + 19)*Acts::units::_mm, 0., 0.});
  }

  // Create plane surfaces without boundaries
  //std::vector<std::shared_ptr<const Acts::PlaneSurface>> surfaces;
  //std::vector<const Acts::Surface*> surfacePtrs;
  //for(unsigned int isur = 0; isur< s_surfacesSize; isur++){
  // surfaces.push_back(std::make_shared<Acts::PlaneSurface>(translations[isur], Acts::Vector3D(1,0,0)));
  //  surfacePtrs.push_back(surfaces[isur].get());
  //}

  std::vector<Acts::PlaneSurface> surfaces;
  std::vector<const Acts::Surface*> surfacePtrs;
  for(unsigned int isur = 0; isur< s_surfacesSize; isur++){
    surfaces.push_back(Acts::PlaneSurface(translations[isur], Acts::Vector3D(1,0,0)));
  }
  for(unsigned int isur = 0; isur< s_surfacesSize; isur++){
    surfacePtrs.push_back(&surfaces[isur]);
  }

  std::cout<<"Creating "<<surfaces.size()<<" boundless plane surfaces"<<std::endl;

  // Create a test context
  GeometryContext gctx;
  MagneticFieldContext mctx;

   InterpolatedBFieldMap3D bField = Options::readBField(bFieldFileName);
   std::cout
      << "Reading BField and creating a 3D InterpolatedBFieldMap instance done"
      << std::endl;

  // Construct a stepper with the bField
  Stepper stepper(bField);
  //Stepper stepper;
  PropagatorType propagator(stepper);
  PropOptionsType propOptions(gctx, mctx);
  propOptions.maxSteps = 10;
  memcpy(propOptions.initializer.surfaceSequence, surfacePtrs.data(), sizeof(const Surface*)*Acts::s_surfacesSize);


  // Construct random starting track parameters
  std::default_random_engine generator(42);
  std::normal_distribution<double> gauss(0., 1.);
  std::uniform_real_distribution<double> unif(-1.0 * M_PI, M_PI);
  std::vector<CurvilinearParameters> startPars;

  for (int i = 0; i < nTracks; i++) {
    BoundSymMatrix cov = BoundSymMatrix::Zero();
    cov << 0.01, 0., 0., 0., 0., 0., 0., 0.01, 0., 0., 0., 0., 0., 0., 0.0001,
        0., 0., 0., 0., 0., 0., 0.0001, 0., 0., 0., 0., 0., 0., 0.0001, 0., 0.,
        0., 0., 0., 0., 1.;

    double q = 1;
    double time = 0;
    double phi = gauss(generator)*0.01;
    double theta = M_PI/2 + gauss(generator)*0.01;
    Vector3D pos(-0, 0.1 * gauss(generator), 0.1 * gauss(generator)); // Units: mm
    Vector3D mom(p*sin(theta)*cos(phi), p*sin(theta)*sin(phi), p*cos(theta)); // Units: GeV 

    startPars.emplace_back(cov, pos, mom, q, time);
  }

  // Propagation result
  PropResultType ress[nTracks];

  auto start = std::chrono::high_resolution_clock::now();

  // Running directly on host or offloading to GPU
  // Run on host
  for (int it = 0; it < nTracks; it++) {
    ress[it] = propagator.propagate(startPars[it], propOptions);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "Time (sec) to run propagation tests: "
            << elapsed_seconds.count() << std::endl;

  if (output) {
    // Write result to obj file
    std::cout << "Writing yielded " << nTracks << " tracks to obj files..."
              << std::endl;

    for (int it = 0; it < nTracks; it++) {
      PropResultType res = ress[it];
      std::ofstream obj_track;
      std::string fileName =
          device + "_output/Track-" + std::to_string(it) + ".obj";
      obj_track.open(fileName.c_str());

      // for (int iv = 0; iv < res.steps(); iv++) {
      //  obj_track << "v " << res.position.col(iv).x() << " "
      //           << res.position.col(iv).y() << " " <<
      //           res.position.col(iv).z()
      //          << std::endl;
      //}
      // for (unsigned int iv = 2; iv <= res.steps(); ++iv) {
      //  obj_track << "l " << iv - 1 << " " << iv << std::endl;
      //}
      
      obj_track.close();
    }
  }

  std::cout << "------------------------  ending  -----------------------"
            << std::endl;

  return 0;
}
