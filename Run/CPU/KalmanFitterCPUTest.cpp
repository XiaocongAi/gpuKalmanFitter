#include "ActsFatras/Kernel/MinimalSimulator.hpp"
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
#include "Utilities/Logger.hpp"
#include "Utilities/ParameterDefinitions.hpp"
#include "Utilities/Units.hpp"
#include "ActsExamples/RandomNumbers.hpp"
#include "Test/Helper.hpp"
#include "Test/Processor.hpp"

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

using Stepper = EigenStepper<Test::ConstantBField>;
// using Stepper = EigenStepper<InterpolatedBFieldMap3D>;
using PropagatorType = Propagator<Stepper>;
using PropResultType = PropagatorResult;
using Simulator = ActsFatras::MinimalSimulator<RandomEngine>;
using PropOptionsType = PropagatorOptions<Simulator, Test::VoidAborter>;
using PlaneSurfaceType = PlaneSurface<InfiniteBounds>;

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

  // Create a random number service
  ActsExamples::RandomNumbers::Config config;
  config.seed = static_cast<uint64_t>(1234567890u);
  auto randomNumbers = std::make_shared<ActsExamples::RandomNumbers>(config);
  auto generator = randomNumbers->spawnGenerator(0);
  std::normal_distribution<double> gauss(0, 1);

  // Create a test context
  GeometryContext gctx;
  MagneticFieldContext mctx;

  // Create the geometry
  size_t nSurfaces = 10;
  // Set translation vectors
  std::vector<Acts::Vector3D> translations;
  for (unsigned int isur = 0; isur < nSurfaces; isur++) {
    translations.push_back({(isur * 30. + 19) * Acts::units::_mm, 0., 0.});
  }
  // Create plane surfaces without boundaries
  std::vector<PlaneSurfaceType> surfaces;
  for (unsigned int isur = 0; isur < nSurfaces; isur++) {
    surfaces.push_back(
        PlaneSurfaceType(translations[isur], Acts::Vector3D(1, 0, 0)));
  }
  const Acts::Surface *surfacePtrs = surfaces.data();
  //PlaneSurfaceType surfaceArrs[nSurfaces];
  //for (unsigned int isur = 0; isur < nSurfaces; isur++) {
  //  surfaceArrs[isur] =
  //      PlaneSurfaceType(translations[isur], Acts::Vector3D(1, 0, 0));
  //  const Acts::Surface *surface = &surfaceArrs[isur];
  //  std::cout << (*surface).center(gctx) << std::endl;
  //}
  std::cout << "Creating " << surfaces.size() << " boundless plane surfaces"
            << std::endl;
  
  // Prepare to run the particles generation
  ActsExamples::GaussianVertexGenerator vertexGen;
  vertexGen.stddev[Acts::eFreePos0] = 1.0 * Acts::units::_mm; 
  vertexGen.stddev[Acts::eFreePos1] = 1.0 * Acts::units::_mm;
  vertexGen.stddev[Acts::eFreePos2] = 1.0 * Acts::units::_mm;
  vertexGen.stddev[Acts::eFreeTime] = 1.0 * Acts::units::_ns;
  ActsExamples::ParametricParticleGenerator::Config pgCfg;
  pgCfg.phiMin = -M_PI;
  pgCfg.phiMin = M_PI;
  pgCfg.etaMin = -1; 
  pgCfg.etaMax = 1; 
  pgCfg.thetaMin = 2 * std::atan(std::exp(-etaMin));
  pgCfg.thetaMax = 2 * std::atan(std::exp(-etaMax));
  pgCfg.pMin = 1.0*Acts::units::_GeV;  
  pgCfg.pMin = 10.0*Acts::units::_GeV;  
  pgCfg.numParticles = 1; 
  ActsExamples::Generator generator = ActsExamples::Generator{ActsExamples::FixedMultiplicityGenerator{10000}, std::move(vertexGen),ActsExamples::ParametricParticleGenerator(pgCfg)};
  // Run the generation to generate particles
  std::vector<ActsFatras::Particle> particles;
  runGeneration(rng, generator, particles); 

  // Prepare to run the simulation
  Stepper stepper;
  PropagatorType propagator(stepper);
  PropOptionsType propOptions(gctx, mctx);
  propOptions.maxSteps = 100;
  propOptions.initializer.surfaceSequence = surfacePtrs;
  propOptions.initializer.surfaceSequenceSize = nSurfaces;
  propOptions.action.generator = &generator;
  std::vector<Simulator::result_type> simResult(nTracks);
  auto start = std::chrono::high_resolution_clock::now();
  // Run the simulation to generate sim hits 
  runSimulation(propagator, propOptions, simResult); 
  auto end_propagate = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end_propagate - start;
  std::cout << "Time (sec) to run propagation tests: "
            << elapsed_seconds.count() << std::endl;
  if (output) {
    std::cout << "writing propagation results" << std::endl;
    Test::writeSimHits(simResult);  
  }

  // The hit smearing resolution
  std::array<double, 2> hitResolution = { 30. * Acts::units::_mm, 30. * Acts::units::_mm}; 
  // Run sim hits smearing to create source links 
  PixelSourceLink sourcelinks[nTracks*nSurfaces];
  runHitSmearing(rng, gctx, simResult, hitResolution, sourcelinks, surfaces, nSurfaces);

  // The particle smearing resolution
  Test::ParticleSmearingParameters seedResolution;
  // Run truth seed smearing to create starting parameters 
  CurvilinearParameters* startPars;  
  runParticleSmearing(rng, gctx, particles, seedResolution, startPars);  

  // Prepare to perform fit to the created tracks
  using RecoStepper = EigenStepper<Test::ConstantBField>;
  using RecoPropagator = Propagator<RecoStepper>;
  using KalmanFitter = KalmanFitter<RecoPropagator, GainMatrixUpdater>;
  using KalmanFitterResult =
      KalmanFitterResult<PixelSourceLink, BoundParameters>;
  using TrackState = typename KalmanFitterResult::TrackStateType;

  // Contruct a KalmanFitter instance
  RecoPropagator rPropagator(stepper);
  KalmanFitter kFitter(rPropagator);
  KalmanFitterOptions<VoidOutlierFinder> kfOptions(gctx, mctx);

  // The fitted results
  std::vector<TrackState> fittedTracks(nSurfaces * nTracks);
  
   int threads = 1;
  auto start_fit = std::chrono::high_resolution_clock::now();
#pragma omp parallel for num_threads(250)
  for (int it = 0; it < nTracks; it++) {
    // The fit result wrapper  
    KalmanFitterResult kfResult;
    kfResult.fittedStates = CudaKernelContainer<TrackState>(
        fittedTracks.data() + it * nSurfaces, nSurfaces);
    // The input source links wrapper
    auto sourcelinkTrack = CudaKernelContainer<PixelSourceLink>(
        sourcelinks+ it*nSurfaces, nSurfaces);
    // Run the fit. The fittedTracks will be changed here
    auto fitStatus = kFitter.fit(sourcelinkTrack, startPars[it], kfOptions,
                                 kfResult, surfacePtrs, nSurfaces);
    if (not fitStatus) {
      std::cout << "fit failure for track " << it << std::endl;
    }
  }
  threads = omp_get_num_threads();
  auto end_fit = std::chrono::high_resolution_clock::now();
  elapsed_seconds = end_fit - start_fit;
  std::cout << "Time (sec) to run KalmanFitter for " << nTracks << " : "
            << elapsed_seconds.count() << std::endl;

  // Log execution time in csv file
  Logger::logTime(Logger::buildFilename("nTracks", std::to_string(nTracks),
                                        "OMP_NumThreads",
                                        std::to_string(threads)),
                  elapsed_seconds.count());

  if (output) {
    std::cout << "writing fitting results" << std::endl;
    Test::writeTracks(fittedTracks.data(), nTracks,nSurfaces);
  }

  // Write the residual and pull of track parameters to ntuple

  std::cout << "------------------------  ending  -----------------------"
            << std::endl;

  return 0;
}
