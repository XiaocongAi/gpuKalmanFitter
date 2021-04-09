#include "FitData.hpp"
#include "Processor.hpp"
#include "Writer.hpp"

#include "Material/HomogeneousSurfaceMaterial.hpp"

#include "ActsExamples/MultiplicityGenerators.hpp"
#include "ActsExamples/ParametricParticleGenerator.hpp"
#include "ActsExamples/VertexGenerators.hpp"

#include "Test/Helper.hpp"
#include "Test/Logger.hpp"

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
            << "\t-o,--output \tIndicator for writing propagation results\n"
            << "\t-r,--threads \tSpecify the number of threads\n"
            << "\t-m,--smoothing \tIndicator for running smoothing\n"
            << "\t-a,--machine \tThe name of the machine, e.g. V100\n"
            << std::endl;
}

int main(int argc, char *argv[]) {
  unsigned int nTracks = 10000;
  unsigned int nThreads = 250;
  bool output = false;
  bool smoothing = true;
  std::string device;
  std::string machine;
  std::string bFieldFileName;
  ActsScalar p;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "-h") or (arg == "--help")) {
      show_usage(argv[0]);
      return 0;
    } else if (i + 1 < argc) {
      if ((arg == "-t") or (arg == "--tracks")) {
        nTracks = atoi(argv[++i]);
      } else if ((arg == "-o") or (arg == "--output")) {
        output = (atoi(argv[++i]) == 1);
      } else if ((arg == "-r") or (arg == "--threads")) {
        nThreads = atoi(argv[++i]);
      } else if ((arg == "-m") or (arg == "--smoothing")) {
        smoothing = (atoi(argv[++i]) == 1);
      } else if ((arg == "-a") or (arg == "--machine")) {
        machine = argv[++i];
      } else {
        std::cerr << "Unknown argument." << std::endl;
        return 1;
      }
    }
  }

  if (machine.empty()) {
    std::cout << "ERROR: The name of the CPU being tested must be provided, "
                 "like e.g. "
                 "-a Intel_i7-8559U."
              << std::endl;
    return 1;
  }

  bool doublePrecision = std::is_same<ActsScalar, double>::value;
  std::cout << "INFO: " << (doublePrecision ? "double" : "float")
            << " precision operand used." << std::endl;

  // Create a random number service
  ActsExamples::RandomNumbers::Config config;
  auto randomNumbers = std::make_shared<ActsExamples::RandomNumbers>(config);
  auto rng = randomNumbers->spawnGenerator(0);

  // Create a test context
  Acts::GeometryContext gctx;
  Acts::MagneticFieldContext mctx;

  // Create the geometry
  size_t nSurfaces = 10;
  // Set translation vectors
  std::vector<Acts::Vector3D> translations;
  for (unsigned int isur = 0; isur < nSurfaces; isur++) {
    Acts::Vector3D translation(isur * 30. + 20., 0., 0.);
    translations.emplace_back(translation);
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
  std::cout << "INFO: Creating " << surfaces.size()
            << " boundless plane surfaces" << std::endl;

  // Assign the geometry ID
  for (Size isur = 0; isur < nSurfaces; isur++) {
    auto geoID = Acts::GeometryID()
                     .setVolume(0u)
                     .setLayer((uint64_t)(isur))
                     .setSensitive((uint64_t)(isur));
    surfaces[isur].assignGeoID(geoID);
  }

  // Prepare to run the particles generation
  ActsExamples::GaussianVertexGenerator vertexGen;
  vertexGen.stddev[Acts::eFreePos0] = 20.0 * Acts::units::_um;
  vertexGen.stddev[Acts::eFreePos1] = 20.0 * Acts::units::_um;
  vertexGen.stddev[Acts::eFreePos2] = 50.0 * Acts::units::_um;
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
  std::cout << std::endl;
  std::cout << "INFO: Time (ms) to run simulation: "
            << elapsed_seconds.count() * 1000 << std::endl;
  if (output) {
    std::string simFileName =
        "sim_hits_for_" + std::to_string(nTracks) + "_particles.obj";
    std::cout << "INFO: Writing simulation results to " << simFileName
              << std::endl;
    writeSimHitsObj(simResult, simFileName);
  }

  // Build the target surfaces based on the truth particle position
  std::vector<Acts::LineSurface> targetSurfaces(nTracks);
  buildTargetSurfaces(validParticles, targetSurfaces.data());

  // The hit smearing resolution
  std::array<ActsScalar, 2> hitResolution = {50. * Acts::units::_um,
                                             50. * Acts::units::_um};
  // Run sim hits smearing to create source links
  std::vector<Acts::PixelSourceLink> sourcelinks(nTracks * nSurfaces);
  // @note pass the concreate PlaneSurfaceType pointer here
  runHitSmearing(gctx, rng, simResult, hitResolution, sourcelinks.data(),
                 surfaces.data(), nSurfaces);

  // The particle smearing resolution
  ParticleSmearingParameters seedResolution;
  // Run truth seed smearing to create starting parameters with provided
  // reference surface
  auto startPars =
      runParticleSmearing(rng, gctx, validParticles, seedResolution,
                          targetSurfaces.data(), nTracks);

  // Prepare to perform fit to the created tracks
  KalmanFitterType kFitter(propagator);
  std::vector<TSType> fittedStates(nSurfaces * nTracks);
  std::vector<Acts::BoundParameters<Acts::LineSurface>> fittedParams(nTracks);
  bool fitStatus[nTracks];

  int threads = 1;
  auto start_fit = std::chrono::high_resolution_clock::now();
#pragma omp parallel for num_threads(nThreads)
  for (int it = 0; it < nTracks; it++) {
    // The fit result wrapper
    KalmanFitterResultType kfResult;
    kfResult.fittedStates = Acts::CudaKernelContainer<TSType>(
        fittedStates.data() + it * nSurfaces, nSurfaces);
    // The input source links wrapper
    auto sourcelinkTrack = Acts::CudaKernelContainer<Acts::PixelSourceLink>(
        sourcelinks.data() + it * nSurfaces, nSurfaces);
    FitOptionsType kfOptions(gctx, mctx, smoothing);
    kfOptions.referenceSurface = &startPars[it].referenceSurface();
    // Run the fit. The fittedStates will be changed here
    auto status = kFitter.fit(sourcelinkTrack, startPars[it], kfOptions,
                              kfResult, surfacePtrs, nSurfaces);
    // if (not status) {
    //  std::cout << "WARNING: fit failure for track " << it << std::endl;
    //}

    // Store the fit parameters and status
    fitStatus[it] = status;
    fittedParams[it] = kfResult.fittedParameters;
    threads = omp_get_num_threads();
  }
  auto end_fit = std::chrono::high_resolution_clock::now();
  elapsed_seconds = end_fit - start_fit;

  // Log the timing measurement in ms
  std::cout << "INFO: Time (ms) to run KF track fitting for " << nTracks
            << " with " << threads
            << " OMP threads: " << elapsed_seconds.count() * 1000 << std::endl;

  // Persistify the timing measurement in ms
  std::string precision = doublePrecision ? "timing_double" : "timing";
  Test::Logger::logTime(
      Test::Logger::buildFilename(precision, machine, "nTracks",
                                  std::to_string(nTracks), "OMP_NumThreads",
                                  std::to_string(threads)),
      elapsed_seconds.count() * 1000);

  if (output) {
    std::cout << "INFO: Writing KF track fitting results" << std::endl;
    std::string state = smoothing ? "smoothed" : "filtered";
    // Write fitted states to obj file
    std::string stateFileName = "fitted_" + state + "_" + machine +
                                "_nTracks_" + std::to_string(nTracks) + ".obj";
    writeStatesObj(fittedStates.data(), fitStatus, nTracks, nSurfaces,
                   stateFileName, state);
    if (smoothing) {
      // Write fitted params to cvs file
      std::string csvFileName = "fitted_param_" + machine + "_nTracks_" +
                                std::to_string(nTracks) + ".csv";
      writeParamsCsv(fittedParams.data(), fitStatus, nTracks, csvFileName);
      // Write fitted params and residual/pull to root file
      std::string rootFileName = "fitted_param_" + machine + "_nTracks_" +
                                 std::to_string(nTracks) + ".root";
      writeParamsRoot(gctx, fittedParams.data(), fitStatus, validParticles,
                      nTracks, rootFileName, "params");
    }
  }

  std::cout << "------------------------  ending  -----------------------"
            << std::endl;

  return 0;
}
