#include "EventData/PixelSourceLink.hpp"
#include "EventData/TrackParameters.hpp"
#include "Fitter/GainMatrixSmoother.hpp"
#include "Fitter/GainMatrixUpdater.hpp"
#include "Fitter/KalmanFitter.hpp"
#include "Material/HomogeneousSurfaceMaterial.hpp"
#include "Propagator/EigenStepper.hpp"
#include "Propagator/Propagator.hpp"
#include "Utilities/ParameterDefinitions.hpp"
#include "Utilities/Units.hpp"

#include "ActsExamples/Generator.hpp"
#include "ActsExamples/MultiplicityGenerators.hpp"
#include "ActsExamples/ParametricParticleGenerator.hpp"
#include "ActsExamples/RandomNumbers.hpp"
#include "ActsExamples/VertexGenerators.hpp"

#include "Test/Helper.hpp"
#include "Test/Logger.hpp"

#include "Processor.hpp"
#include "Writer.hpp"

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
            //<< "\t-p,--pt \tSpecify the pt of particle\n"
            << "\t-o,--output \tIndicator for writing propagation results\n"
            //<< "\t-b,--bf-map \tSpecify the path of *.txt for interpolated "
            //   "BField map\n"
            << "\t-r,--threads \tSpecify the number of threads\n"
            << std::endl;
}

using Stepper = Acts::EigenStepper<Test::ConstantBField>;
using PropagatorType = Acts::Propagator<Stepper>;
using PropResultType = Acts::PropagatorResult;
using PropOptionsType = Acts::PropagatorOptions<Simulator, Test::VoidAborter>;
using Smoother = GainMatrixSmoother<BoundParameters>;
using KalmanFitterType =
    Acts::KalmanFitter<PropagatorType, Acts::GainMatrixUpdater, Smoother>;
using KalmanFitterResultType =
    Acts::KalmanFitterResult<Acts::PixelSourceLink, Acts::BoundParameters>;
using TSType = typename KalmanFitterResultType::TrackStateType;

int main(int argc, char *argv[]) {
  unsigned int nTracks = 10000;
  unsigned int nThreads = 250;
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
        //} else if ((arg == "-p") or (arg == "--pt")) {
        //  p = atof(argv[++i]) * Acts::units::_GeV;
      } else if ((arg == "-o") or (arg == "--output")) {
        output = (atoi(argv[++i]) == 1);
      } else if ((arg == "-r") or (arg == "--threads")) {
        nThreads = atoi(argv[++i]);
      } else {
        std::cerr << "Unknown argument." << std::endl;
        return 1;
      }
    }
  }

  // Create a random number service
  ActsExamples::RandomNumbers::Config config;
  auto randomNumbers = std::make_shared<ActsExamples::RandomNumbers>(config);
  auto rng = randomNumbers->spawnGenerator(0);

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
  // The silicon material
  Acts::MaterialSlab matProp(Test::makeSilicon(), 0.5 * Acts::units::_mm);
  Acts::HomogeneousSurfaceMaterial surfaceMaterial(matProp);
  // Create plane surfaces without boundaries
  std::vector<PlaneSurfaceType> surfaces;
  for (unsigned int isur = 0; isur < nSurfaces; isur++) {
    surfaces.push_back(PlaneSurfaceType(
        translations[isur], Acts::Vector3D(1, 0, 0), surfaceMaterial));
  }
  const Acts::Surface *surfacePtrs = surfaces.data();
  std::cout << "Creating " << surfaces.size() << " boundless plane surfaces"
            << std::endl;

  // Prepare to run the particles generation
  ActsExamples::GaussianVertexGenerator vertexGen;
  vertexGen.stddev[Acts::eFreePos0] = 1.0 * Acts::units::_mm;
  vertexGen.stddev[Acts::eFreePos1] = 1.0 * Acts::units::_mm;
  vertexGen.stddev[Acts::eFreePos2] = 5.0 * Acts::units::_mm;
  vertexGen.stddev[Acts::eFreeTime] = 1.0 * Acts::units::_ns;
  ActsExamples::ParametricParticleGenerator::Config pgCfg;
  // @note We are generating 20% more particles to make sure we could get enough
  // valid particles
  size_t nGeneratedParticles = nTracks * 1.2;
  ActsExamples::Generator generator = ActsExamples::Generator{
      ActsExamples::FixedMultiplicityGenerator{nGeneratedParticles},
      std::move(vertexGen), ActsExamples::ParametricParticleGenerator(pgCfg)};
  // Run the generation to generate particles
  std::vector<ActsFatras::Particle> generatedParticles;
  runParticleGeneration(rng, generator, generatedParticles);

  // Prepare to run the simulation
  Stepper stepper;
  PropagatorType propagator(stepper);
  std::vector<ActsFatras::Particle> validParticles(nTracks);
  std::vector<Simulator::result_type> simResult(nTracks);
  auto start_propagate = std::chrono::high_resolution_clock::now();
  // Run the simulation to generate sim hits
  // @note We will pick up the valid particles
  runSimulation(gctx, mctx, rng, propagator, generatedParticles, validParticles,
                simResult, surfacePtrs, nSurfaces);
  auto end_propagate = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds =
      end_propagate - start_propagate;
  std::cout << "Time (ms) to run propagation tests: "
            << elapsed_seconds.count() * 1000 << std::endl;
  if (output) {
    std::cout << "writing propagation results" << std::endl;
    writeSimHitsObj(simResult);
  }

  // The hit smearing resolution
  std::array<double, 2> hitResolution = {30. * Acts::units::_mm,
                                         30. * Acts::units::_mm};
  // Run sim hits smearing to create source links
  Acts::PixelSourceLink sourcelinks[nTracks * nSurfaces];
  // @note pass the concreate PlaneSurfaceType pointer here
  runHitSmearing(gctx, rng, simResult, hitResolution, sourcelinks,
                 surfaces.data(), nSurfaces);

  // The particle smearing resolution
  ParticleSmearingParameters seedResolution;
  // Run truth seed smearing to create starting parameters
  auto startPars = runParticleSmearing(rng, gctx, generatedParticles,
                                       seedResolution, nTracks);

  // Prepare to perform fit to the created tracks
  KalmanFitterType kFitter(propagator);
  std::vector<TSType> fittedStates(nSurfaces * nTracks);
  std::vector<Acts::BoundParameters> fittedParams(nTracks);
  bool fitStatus[nTracks];

  int threads = 1;
  auto start_fit = std::chrono::high_resolution_clock::now();
  std::cout << " Run the fit" << std::endl;
#pragma omp parallel for num_threads(nThreads)
  for (int it = 0; it < nTracks; it++) {
    // The fit result wrapper
    KalmanFitterResultType kfResult;
    kfResult.fittedStates = CudaKernelContainer<TSType>(
        fittedStates.data() + it * nSurfaces, nSurfaces);
    // The input source links wrapper
    auto sourcelinkTrack = CudaKernelContainer<PixelSourceLink>(
        sourcelinks + it * nSurfaces, nSurfaces);
    // @todo Use perigee surface as the target surface. Needs a perigee surface
    // object
    KalmanFitterOptions<Acts::VoidOutlierFinder> kfOptions(gctx, mctx);
    kfOptions.referenceSurface = &startPars[it].referenceSurface();
    // @note when it >=35, we got different startPars[i] between CPU and GPU
    // Run the fit. The fittedStates will be changed here
    auto status = kFitter.fit(sourcelinkTrack, startPars[it], kfOptions,
                              kfResult, surfacePtrs, nSurfaces);
    if (not status) {
      std::cout << "fit failure for track " << it << std::endl;
    }
    // store the fit parameters and status
    fitStatus[it] = status;
    fittedParams[it] = kfResult.fittedParameters;
    threads = omp_get_num_threads();
  }
  std::cout << "threads = " << threads << std::endl;
  auto end_fit = std::chrono::high_resolution_clock::now();
  elapsed_seconds = end_fit - start_fit;
  std::cout << "Time (ms) to run KalmanFitter for " << nTracks << " : "
            << elapsed_seconds.count() * 1000 << std::endl;

  // Log execution time in csv file
  Test::Logger::logTime(Test::Logger::buildFilename(
                            "timing_cpu", "nTracks", std::to_string(nTracks),
                            "OMP_NumThreads", std::to_string(threads)),
                        elapsed_seconds.count());

  if (output) {
    std::cout << "writing fitting results" << std::endl;
    std::string param = "smoothed";
    std::string stateFileName =
        "fitted_" + param + "_cpu_nTracks_" + std::to_string(nTracks) + ".obj";
    writeStatesObj(fittedStates.data(), fitStatus, nTracks, nSurfaces,
                stateFileName, param);
    std::string paramFileName =
        "fitted_param_cpu_nTracks_" + std::to_string(nTracks) + ".csv";
    writeParamsCsv(fittedParams.data(), fitStatus, nTracks, paramFileName);
  }

  // @todo Write the residual and pull of track parameters to ntuple

  std::cout << "------------------------  ending  -----------------------"
            << std::endl;

  return 0;
}
