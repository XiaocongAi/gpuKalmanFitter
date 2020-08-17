#include "EventData/PixelSourceLink.hpp"
#include "EventData/TrackParameters.hpp"
#include "Fitter/GainMatrixUpdater.hpp"
#include "Fitter/KalmanFitter.hpp"
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
            //<< "\t-d,--device \tSpecify the device: 'gpu' or 'cpu'\n"
            //<< "\t-b,--bf-map \tSpecify the path of *.txt for interpolated "
            //   "BField map\n"
            << std::endl;
}

using namespace Acts;

std::default_random_engine generator(42);
std::normal_distribution<double> gauss(0., 1.);
std::uniform_real_distribution<double> unif(-1.0 * M_PI, M_PI);

// Struct for B field
struct ConstantBField {
  ACTS_DEVICE_FUNC static Vector3D getField(const Vector3D & /*field*/) {
    return Vector3D(0., 0., 2.0 * Acts::units::_T);
  }
};

// Measurement creator
struct MeasurementCreator {
  double resX = 30 * Acts::units::_um;
  double resY = 30 * Acts::units::_um;

  struct this_result {
    std::vector<PixelSourceLink> sourcelinks;
  };
  using result_type = this_result;

  template <typename propagator_state_t, typename stepper_t>
  void operator()(propagator_state_t &state, const stepper_t &stepper,
                  result_type &result) const {
    if (state.navigation.currentSurface != nullptr) {

      // Apply global to local
      Vector2D lPos;
      state.navigation.currentSurface->globalToLocal(
          state.options.geoContext, stepper.position(state.stepping),
          stepper.direction(state.stepping), lPos);
      // Perform the smearing to truth
      double dx = resX * gauss(generator);
      double dy = resY * gauss(generator);

      // The measurement values
      Vector2D values;
      values << lPos[0] + dx, lPos[1] + dy;

      // The measurement covariance
      SymMatrix2D cov;
      cov << resX * resX, 0., 0., resY * resY;

      // Push back to the container
      result.sourcelinks.emplace_back(values, cov,
                                      state.navigation.currentSurface);
    }
    return;
  }
};

// Test actor
struct VoidActor {
  struct this_result {};
  using result_type = this_result;

  template <typename propagator_state_t, typename stepper_t>
  void operator()(propagator_state_t &state, const stepper_t &stepper,
                  result_type &result) const {
    if (state.navigation.currentSurface != nullptr) {
      std::cout << "On surface: " << state.navigation.nextSurfaceIter
                << std::endl;
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

using Stepper = EigenStepper<ConstantBField>;
// using Stepper = EigenStepper<InterpolatedBFieldMap3D>;
using PropagatorType = Propagator<Stepper>;
using PropResultType = PropagatorResult;
using PropOptionsType = PropagatorOptions<MeasurementCreator, VoidAborter>;

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
        p = atof(argv[++i]) * Acts::units::_GeV;
      } else if ((arg == "-o") or (arg == "--output")) {
        output = (atoi(argv[++i]) == 1);
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
  GeometryContext gctx;
  MagneticFieldContext mctx;

  // Create the geometry
  size_t nSurfaces = 15;

  // Set translation vectors
  std::vector<Acts::Vector3D> translations;
  for (unsigned int isur = 0; isur < nSurfaces; isur++) {
    translations.push_back({(isur * 30. + 19) * Acts::units::_mm, 0., 0.});
  }

  // Create plane surfaces without boundaries
  std::vector<Acts::PlaneSurface> surfaces;
  for (unsigned int isur = 0; isur < nSurfaces; isur++) {
    surfaces.push_back(
        Acts::PlaneSurface(translations[isur], Acts::Vector3D(1, 0, 0)));
  }

  const Acts::Surface *surfacePtrs[nSurfaces];
  for (unsigned int isur = 0; isur < nSurfaces; isur++) {
    surfacePtrs[isur] = &surfaces[isur];
  }

  Acts::PlaneSurface surfaceArrs[nSurfaces];
  for (unsigned int isur = 0; isur < nSurfaces; isur++) {
    surfaceArrs[isur] =
        Acts::PlaneSurface(translations[isur], Acts::Vector3D(1, 0, 0));
    const Acts::Surface *surface = &surfaceArrs[isur];
    std::cout << (*surface).center(gctx) << std::endl;
  }

  std::cout << "Creating " << surfaces.size() << " boundless plane surfaces"
            << std::endl;

  //  // Test the pointers to surfaces
  //  const PlaneSurface *surfacePtr = surfaces.data();
  //  for (unsigned int isur = 0; isur < nSurfaces; isur++) {
  //     std::cout<<"surface " << isur <<  " has center at: \n"
  //     <<(*surfacePtr).center(gctx)<<std::endl;
  //    surfacePtr++;
  //  }

  //   InterpolatedBFieldMap3D bField = Options::readBField(bFieldFileName);
  //   std::cout
  //      << "Reading BField and creating a 3D InterpolatedBFieldMap instance
  //      done"
  //      << std::endl;

  // Construct a stepper with the bField
  // Stepper stepper(bField);
  Stepper stepper;
  PropagatorType propagator(stepper);
  PropOptionsType propOptions(gctx, mctx);
  propOptions.maxSteps = 100;
  propOptions.initializer.surfaceSequence = surfacePtrs;
  propOptions.initializer.surfaceSequenceSize = nSurfaces;

  // Construct random starting track parameters
  std::vector<CurvilinearParameters> startPars;
  double resLoc1 = 0.1 * Acts::units::_mm;
  double resLoc2 = 0.1 * Acts::units::_mm;
  double resPhi = 0.01;
  double resTheta = 0.01;
  for (int i = 0; i < nTracks; i++) {
    BoundSymMatrix cov = BoundSymMatrix::Zero();
    cov << resLoc1 * resLoc1, 0., 0., 0., 0., 0., 0., resLoc2 * resLoc2, 0., 0.,
        0., 0., 0., 0., resPhi * resPhi, 0., 0., 0., 0., 0., 0.,
        resTheta * resTheta, 0., 0., 0., 0., 0., 0., 0.0001, 0., 0., 0., 0., 0.,
        0., 1.;

    double q = 1;
    double time = 0;
    double phi = gauss(generator) * resPhi;
    double theta = M_PI / 2 + gauss(generator) * resTheta;
    Vector3D pos(0, resLoc1 * gauss(generator),
                 resLoc2 * gauss(generator)); // Units: mm
    Vector3D mom(p * sin(theta) * cos(phi), p * sin(theta) * sin(phi),
                 p * cos(theta)); // Units: GeV

    startPars.emplace_back(cov, pos, mom, q, time);
  }

  // Propagation result
  MeasurementCreator::result_type ress[nTracks];

  auto start = std::chrono::high_resolution_clock::now();

  // Creating the tracks
  for (int it = 0; it < nTracks; it++) {
    propagator.propagate(startPars[it], propOptions, ress[it]);
  }

  auto end_propagate = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end_propagate - start;
  std::cout << "Time (sec) to run propagation tests: "
            << elapsed_seconds.count() << std::endl;

  unsigned int vCounter = 0;
  if (output) {
    // Write result to obj file

    // Write one track to one obj file
    /*
    for (int it = 0; it < nTracks; it++) {
      auto tracks = ress[it].actorResult.sourcelinks;
      std::ofstream obj_track;
      std::string fileName ="cpu_output/Track-" + std::to_string(it) + ".obj";
      obj_track.open(fileName.c_str());

       for (const auto& sl: tracks) {
        const auto& pos = sl.globalPosition(gctx);
        obj_track << "v " << pos.x() << " "
                 << pos.y() << " " <<
                 pos.z()
                << std::endl;
      }
       for (unsigned int iv = 2; iv <= tracks.size(); ++iv) {
        obj_track << "l " << iv - 1 << " " << iv << std::endl;
      }

      obj_track.close();
    }
    */

    std::cout << "writing propagation results" << std::endl;
    // Write all of the created tracks to one obj file
    std::ofstream obj_track;
    std::string fileName = "Tracks-propagation-gcc.obj";
    obj_track.open(fileName.c_str());

    // Initialize the vertex counter
    for (int it = 0; it < nTracks; it++) {
      auto tracks = ress[it].sourcelinks;
      ++vCounter;
      for (const auto &sl : tracks) {
        const auto &pos = sl.globalPosition(gctx);
        obj_track << "v " << pos.x() << " " << pos.y() << " " << pos.z()
                  << "\n";
      }
      // Write out the line - only if we have at least two points created
      size_t vBreak = vCounter + tracks.size() - 1;
      for (; vCounter < vBreak; ++vCounter)
        obj_track << "l " << vCounter << " " << vCounter + 1 << '\n';
    }
    obj_track.close();
  }

  // start to perform fit to the created tracks
  using RecoStepper = EigenStepper<ConstantBField>;
  using RecoPropagator = Propagator<RecoStepper>;
  using KalmanFitter = KalmanFitter<RecoPropagator, GainMatrixUpdater>;
  using KalmanFitterResult =
      KalmanFitterResult<PixelSourceLink, BoundParameters>;
  using TrackState = typename KalmanFitterResult::TrackStateType;

  // Contruct a KalmanFitter instance
  RecoPropagator rPropagator(stepper);
  KalmanFitter kFitter(rPropagator);
  KalmanFitterOptions<VoidOutlierFinder> kfOptions(gctx, mctx);

  std::vector<TrackState> fittedTracks(nSurfaces * nTracks);

  for (int it = 0; it < nTracks; it++) {
    BoundSymMatrix cov = BoundSymMatrix::Zero();
    cov << resLoc1 * resLoc1, 0., 0., 0., 0., 0., 0., resLoc2 * resLoc2, 0., 0.,
        0., 0., 0., 0., resPhi * resPhi, 0., 0., 0., 0., 0., 0.,
        resTheta * resTheta, 0., 0., 0., 0., 0., 0., 0.0001, 0., 0., 0., 0., 0.,
        0., 1.;

    double q = 1;
    double time = 0;
    Vector3D pos(0, 0, 0); // Units: mm
    Vector3D mom(p, 0, 0); // Units: GeV

    CurvilinearParameters rStart(cov, pos, mom, q, time);

    KalmanFitterResult kfResult;
    kfResult.fittedStates = CudaKernelContainer<TrackState>(
        fittedTracks.data() + it * nSurfaces, nSurfaces);

    auto sourcelinkTrack = CudaKernelContainer<PixelSourceLink>(
        ress[it].sourcelinks.data(), ress[it].sourcelinks.size());
    // The fittedTracks will be changed here
    auto fitStatus = kFitter.fit(sourcelinkTrack, rStart, kfOptions, kfResult,
                                 surfacePtrs, nSurfaces);
    if (not fitStatus) {
      std::cout << "fit failure for track " << it << std::endl;
    }
  }

  auto end_fit = std::chrono::high_resolution_clock::now();
  elapsed_seconds = end_fit - end_propagate;
  std::cout << "Time (sec) to run KalmanFitter for " << nTracks << " : "
            << elapsed_seconds.count() << std::endl;

  if (output) {
    std::cout << "writing KF results" << std::endl;
    // Write all of the created tracks to one obj file
    std::ofstream obj_ftrack;
    std::string fileName_ = "Tracks-fitted-gcc.obj";
    obj_ftrack.open(fileName_.c_str());

    // Initialize the vertex counter
    vCounter = 0;
    for (int it = 0; it < nTracks; it++) {
      ++vCounter;
      for (int is = 0; is < nSurfaces; is++) {
        const auto &pos =
            fittedTracks[it * nSurfaces + is].parameter.filtered.position();
        obj_ftrack << "v " << pos.x() << " " << pos.y() << " " << pos.z()
                   << "\n";
      }
      // Write out the line - only if we have at least two points created
      size_t vBreak = vCounter + nSurfaces - 1;
      for (; vCounter < vBreak; ++vCounter)
        obj_ftrack << "l " << vCounter << " " << vCounter + 1 << '\n';
    }
    obj_ftrack.close();
  }

  std::cout << "------------------------  ending  -----------------------"
            << std::endl;

  return 0;
}
