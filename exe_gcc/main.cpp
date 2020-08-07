#include "EventData/TrackParameters.hpp"
#include "Geometry/GeometryContext.hpp"
#include "MagneticField/MagneticFieldContext.hpp"
#include "Plugins/BFieldOptions.hpp"
#include "Plugins/BFieldUtils.hpp"
#include "Propagator/EigenStepper.hpp"
#include "Propagator/Propagator.hpp"
#include "Utilities/ParameterDefinitions.hpp"

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
    return Vector3D(0., 0., 2.);
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

using Stepper = EigenStepper<ConstantBField>;
// using Stepper = EigenStepper<InterpolatedBFieldMap3D>;
using PropagatorType = Propagator<Stepper>;
using PropResultType =
    PropagatorResult<typename VoidActor::result_type>;
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
  double pT;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "-h") or (arg == "--help")) {
      show_usage(argv[0]);
      return 0;
    } else if (i + 1 < argc) {
      if ((arg == "-t") or (arg == "--tracks")) {
        nTracks = atoi(argv[++i]);
      } else if ((arg == "-p") or (arg == "--pt")) {
        pT = atof(argv[++i]);
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

  // Create a test context
  GeometryContext gctx = GeometryContext();
  MagneticFieldContext mctx = MagneticFieldContext();

  // InterpolatedBFieldMap3D bField = Options::readBField(bFieldFileName);
  // std::cout
  //    << "Reading BField and creating a 3D InterpolatedBFieldMap instance
  //    done"
  //    << std::endl;

  // Construct a stepper with the bField
  // Stepper stepper(bField);
  Stepper stepper;
  // Construct a propagator
  PropagatorType propagator(stepper);
  // Construct the propagation options object
  PropOptionsType propOptions(gctx, mctx);
  propOptions.maxSteps = 10;

  // Construct random starting track parameters
  std::default_random_engine generator(42);
  std::normal_distribution<double> gauss(0., 1.);
  std::uniform_real_distribution<double> randPhi(-1.0 * M_PI, M_PI);
  std::uniform_real_distribution<double> randTheta(0, M_PI);
  // CurvilinearParameters startPars[nTracks];
  std::vector<CurvilinearParameters> startPars;

  for (int i = 0; i < nTracks; i++) {
    BoundSymMatrix cov = BoundSymMatrix::Zero();
    cov << 0.01, 0., 0., 0., 0., 0., 0., 0.01, 0., 0., 0., 0., 0., 0., 0.0001,
        0., 0., 0., 0., 0., 0., 0.0001, 0., 0., 0., 0., 0., 0., 0.0001, 0., 0.,
        0., 0., 0., 0., 1.;

    double q = 1;
    double time = 0;
    Vector3D pos(0, 0.1 * gauss(generator),
                 0.1 * gauss(generator)); // Units: mm
    double phi = randPhi(generator);
    double theta = randTheta(generator);
    Vector3D mom(1, 0, 0); // Units: GeV
    // CurvilinearParameters rStart(cov, pos, mom, q, time);
    // startPars[i] = rStart;

    startPars.emplace_back(cov, pos, mom, q, time);
    // std::cout << " rPos = (" << pars[i].position().x() << ", "
    //           << pars[i].position().y() << ", " << pars[i].position().z()
    //           << ") " << std::endl;
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
