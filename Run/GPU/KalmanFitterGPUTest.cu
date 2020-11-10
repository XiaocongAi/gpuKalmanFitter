#include "EventData/PixelSourceLink.hpp"
#include "EventData/TrackParameters.hpp"
#include "Fitter/GainMatrixUpdater.hpp"
#include "Fitter/KalmanFitter.hpp"
#include "Propagator/EigenStepper.hpp"
#include "Propagator/Propagator.hpp"
#include "Utilities/CudaHelper.hpp"
#include "Utilities/Logger.hpp"
#include "Utilities/ParameterDefinitions.hpp"
#include "Utilities/Profiling.hpp"
#include "Utilities/Units.hpp"

#include "ActsExamples/Generator.hpp"
#include "ActsExamples/MultiplicityGenerators.hpp"
#include "ActsExamples/ParametricParticleGenerator.hpp"
#include "ActsExamples/RandomNumbers.hpp"
#include "ActsExamples/VertexGenerators.hpp"
#include "Processor.hpp"
#include "Test/Helper.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// This executable is used to run the KalmanFitter fit test on GPU with
// parallelism on the track-level. It contains mainly two parts: 1) Explicit
// calling of the propagation to create measurements on tracks ( a 'simulated'
// track could contain 10~100 measurements) 2) Running the Kalmanfitter using
// the created measurements in 1) as one of the inputs In princinple, both 1)
// and 2) could on offloaded to GPU. Right now, only 2) is put into a kernel

static void show_usage(std::string name) {
  std::cerr << "Usage: <option(s)> VALUES"
            << "Options:\n"
            << "\t-h,--help\t\tShow this help message\n"
            << "\t-t,--tracks \tSpecify the number of tracks\n"
            << "\t-p,--pt \tSpecify the pt of particle\n"
            << "\t-o,--output \tIndicator for writing propagation results\n"
            << "\t-d,--device \tSpecify the device: 'gpu' or 'cpu'\n"
            << "\t-g,--grid-size \tSpecify GPU grid size: 'x*y'\n"
            << "\t-b,--block-size \tSpecify GPU block size: 'x*y*z'\n"
            << "\t-s,--shared-memory \tIndicator for using shared memory for "
               "one track or not\n"
            << std::endl;
}

using Stepper = Acts::EigenStepper<Test::ConstantBField>;
using PropagatorType = Acts::Propagator<Stepper>;
using PropResultType = Acts::PropagatorResult;
using PropOptionsType = Acts::PropagatorOptions<Simulator, Test::VoidAborter>;
using PropState = PropagatorType::State<PropOptionsType>;
using KalmanFitterType =
    Acts::KalmanFitter<PropagatorType, Acts::GainMatrixUpdater>;
using KalmanFitterResultType =
    Acts::KalmanFitterResult<Acts::PixelSourceLink, Acts::BoundParameters>;
using TSType = typename KalmanFitterResultType::TrackStateType;

// Device code
__global__ void __launch_bounds__(256, 2)
    fitKernelThreadPerTrack(KalmanFitterType *kFitter,
                            Acts::PixelSourceLink *sourcelinks,
                            Acts::CurvilinearParameters *tpars,
                            Acts::KalmanFitterOptions<Acts::VoidOutlierFinder> kfOptions,
                            TSType *fittedTracks, const Acts::Surface *surfacePtrs,
                            int nSurfaces, int nTracks, int offset) {
  // In case of 1D grid and 1D block, the threadId = blockDim.x*blockIdx.x +
  // threadIdx.x + offset
  int threadId =
      blockDim.x * blockDim.y * (gridDim.x * blockIdx.y + blockIdx.x) +
      blockDim.x * threadIdx.y + threadIdx.x + offset;

  // Different threads handles different track
  if (threadId < (nTracks + offset)) {
    // Use the CudaKernelContainer for the source links and fitted tracks
    KalmanFitterResultType kfResult;
    kfResult.fittedStates = CudaKernelContainer<TSType>(
        fittedTracks + threadId * nSurfaces, nSurfaces);
    kFitter->fit(Acts::CudaKernelContainer<PixelSourceLink>(
                     sourcelinks + threadId * nSurfaces, nSurfaces),
                 tpars[threadId], kfOptions, kfResult, surfacePtrs, nSurfaces);
  }
}

__global__ void __launch_bounds__(256, 2)
    fitKernelBlockPerTrack(KalmanFitterType *kFitter,
                           Acts::PixelSourceLink *sourcelinks,
                           Acts::CurvilinearParameters *tpars,
                           Acts::KalmanFitterOptions<Acts::VoidOutlierFinder> kfOptions,
                           TSType *fittedTracks, const Acts::Surface *surfacePtrs,
                           int nSurfaces, int nTracks, int offset) {
  int blockId = gridDim.x * blockIdx.y + blockIdx.x + offset;

  // All threads in this block handles the same track
  if (blockId < (nTracks + offset)) {
    // Use the CudaKernelContainer for the source links and fitted tracks
    KalmanFitterResultType kfResult;
    kfResult.fittedStates = CudaKernelContainer<TSType>(
        fittedTracks + blockId * nSurfaces, nSurfaces);
    kFitter->fitOnDevice(Acts::CudaKernelContainer<PixelSourceLink>(
                             sourcelinks + blockId * nSurfaces, nSurfaces),
                         tpars[blockId], kfOptions, kfResult, surfacePtrs,
                         nSurfaces);
  }
}

int main(int argc, char *argv[]) {
  unsigned int nTracks = 10240;
  bool output = false;
  bool useSharedMemory = false;
  std::string device = "cpu";
  std::string bFieldFileName;
  double p = 1 * Acts::units::_GeV;
  dim3 grid(40), block(8, 8);
  bool multipleScattering = false;
  bool energyLoss = false;
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
      } else if ((arg == "-d") or (arg == "--device")) {
        device = argv[++i];
      } else if ((arg == "-g") or (arg == "--grid-size")) {
        grid = stringToDim3(argv[++i]);
      } else if ((arg == "-b") or (arg == "--block-size")) {
        block = stringToDim3(argv[++i]);
      } else if ((arg == "-s") or (arg == "--shared-memory")) {
        useSharedMemory = (atoi(argv[++i]) == 1);
      } else if ((arg == "-m") or (arg == "--multiple-scattering")) {
        multipleScattering = (atoi(argv[++i]) == 1);
      } else if ((arg == "-e") or (arg == "--energy-loss")) {
        energyLoss = (atoi(argv[++i]) == 1);
      } else {
        std::cerr << "Unknown argument." << std::endl;
        return 1;
      }
    }
  }

  if (grid.z != 1 or block.z != 1) {
    std::cout << "3D grid or block is not supported at the moment! Good luck!"
              << std::endl;
    return 1;
  }
  std::cout << grid.x << " " << grid.y << " " << block.x << " " << block.y
            << std::endl;

  int devId = 0;

  cudaDeviceProp prop;
  GPUERRCHK(cudaGetDeviceProperties(&prop, devId));
  printf("Device : %s\n", prop.name);
  GPUERRCHK(cudaSetDevice(devId));
  int driverVersion, rtVersion;
  GPUERRCHK(cudaDriverGetVersion(&driverVersion));
  printf("cuda driver version: %i\n", driverVersion);
  GPUERRCHK(cudaRuntimeGetVersion(&rtVersion));
  printf("cuda rt version: %i\n", rtVersion);

  int tracksPerBlock = block.x * block.y;

  // Use 8*8 block if using one block for one track
  // @todo Extend to run multiple (block.z) tracks in one block
  if (useSharedMemory) {
    std::cout << "Shared memory used. Block size is set to 8*8!" << std::endl;
    block = dim3(8, 8);
    tracksPerBlock = 1;
  }

  const int nStreams = 4;
  // The last stream could could less tracks
  const int tracksPerStream = (nTracks + nStreams - 1) / nStreams;
  const int overflowTracks = tracksPerStream * nStreams - nTracks;
  const int tracksLastStream = tracksPerStream - overflowTracks;
  std::cout << "tracksPerStream = " << tracksPerStream << std::endl;
  std::cout << "tracksLastStream = " << tracksLastStream << std::endl;

  // @note shall we use this for the grid size?
  const int blocksPerGrid_multiStream =
      (tracksPerStream + tracksPerBlock - 1) / tracksPerBlock;

  // The shared memory size
  int sharedMemoryPerTrack = sizeof(PathLimitReached) + sizeof(PropState) +
                             sizeof(bool) * 2 + sizeof(PropagatorResult);
  std::cout << "shared memory is " << sharedMemoryPerTrack << std::endl;

  // The number of test surfaces
  size_t nSurfaces = 10;
  const int surfaceBytes = sizeof(PlaneSurfaceType) * nSurfaces;
  const int sourcelinksBytes = sizeof(PixelSourceLink) * nSurfaces * nTracks;
  const int parsBytes = sizeof(CurvilinearParameters) * nTracks;
  const int tsBytes = sizeof(TSType) * nSurfaces * nTracks;
  std::cout << "surface Bytes = " << surfaceBytes << std::endl;
  std::cout << "source links Bytes = " << sourcelinksBytes << std::endl;
  std::cout << "startPars Bytes = " << parsBytes << std::endl;
  std::cout << "TSs Bytes = " << tsBytes << std::endl;

  const int perStreamSourcelinksBytes =
      sizeof(PixelSourceLink) * nSurfaces * tracksPerStream;
  const int lastStreamSourcelinksBytes =
      sizeof(PixelSourceLink) * nSurfaces * tracksLastStream;
  const int perStreamParsBytes =
      sizeof(CurvilinearParameters) * tracksPerStream;
  const int lastStreamParsBytes =
      sizeof(CurvilinearParameters) * tracksLastStream;
  const int perStreamTSsBytes = sizeof(TSType) * nSurfaces * tracksPerStream;
  const int lastStreamTSsBytes = sizeof(TSType) * nSurfaces * tracksLastStream;

  // Create a test context
  Acts::GeometryContext gctx(0);
  Acts::MagneticFieldContext mctx(0);

  // Create a random number service
  ActsExamples::RandomNumbers::Config config;
  auto randomNumbers = std::make_shared<ActsExamples::RandomNumbers>(config);
  auto rng = randomNumbers->spawnGenerator(0);

  // Create the geometry
  // Set translation vectors
  std::vector<Acts::Vector3D> translations;
  for (unsigned int isur = 0; isur < nSurfaces; isur++) {
    translations.push_back({(isur * 30. + 19) * Acts::units::_mm, 0., 0.});
  }
  // Create plane surfaces without boundaries
  PlaneSurfaceType *surfaces;
  // Unified memory allocation for geometry
  GPUERRCHK(cudaMallocManaged(&surfaces, sizeof(PlaneSurfaceType) * nSurfaces));
  std::cout << "Allocating the memory for the surfaces" << std::endl;
  for (unsigned int isur = 0; isur < nSurfaces; isur++) {
    surfaces[isur] =
        PlaneSurfaceType(translations[isur], Acts::Vector3D(1, 0, 0));
  }
  const Acts::Surface *surfacePtrs = surfaces;
  std::cout << "Creating " << nSurfaces << " boundless plane surfaces"
            << std::endl;

  // Prepare to run the particles generation
  ActsExamples::GaussianVertexGenerator vertexGen;
  vertexGen.stddev[Acts::eFreePos0] = 1.0 * Acts::units::_mm;
  vertexGen.stddev[Acts::eFreePos1] = 1.0 * Acts::units::_mm;
  vertexGen.stddev[Acts::eFreePos2] = 5.0 * Acts::units::_mm;
  vertexGen.stddev[Acts::eFreeTime] = 1.0 * Acts::units::_ns;
  ActsExamples::ParametricParticleGenerator::Config pgCfg;
  ActsExamples::Generator generator = ActsExamples::Generator{
      ActsExamples::FixedMultiplicityGenerator{nTracks}, std::move(vertexGen),
      ActsExamples::ParametricParticleGenerator(pgCfg)};
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
  propOptions.action.generator = &rng;
  std::vector<Simulator::result_type> simResult(nTracks);
  auto start_propagate = std::chrono::high_resolution_clock::now();
  std::cout<<"start to run propagation"<<std::endl; 
 // Run the simulation to generate sim hits
  runSimulation(propagator, propOptions, particles, simResult);
  auto end_propagate = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end_propagate - start_propagate;
  std::cout << "Time (sec) to run propagation tests: "
            << elapsed_seconds.count() << std::endl;
  if (output) {
    std::cout << "writing propagation results" << std::endl;
    Test::writeSimHits(simResult);
  }

  // The hit smearing resolution
  std::array<double, 2> hitResolution = {30. * Acts::units::_mm,
                                         30. * Acts::units::_mm};
  // Pinned memory for source links
  Acts::PixelSourceLink *sourcelinks;
  GPUERRCHK(cudaMallocHost((void **)&sourcelinks, sourcelinksBytes));
  // Run hit smearing to create source links 
  // @note pass the concreate PlaneSurfaceType pointer here 
  runHitSmearing(rng, gctx, simResult, hitResolution, sourcelinks, surfaces,
                 nSurfaces);

  // The particle smearing resolution
  ParticleSmearingParameters seedResolution;
  // Run truth seed smearing to create starting parameters
  auto startParsCollection =
      runParticleSmearing(rng, gctx, particles, seedResolution, nTracks);
  // Pinned memory for starting track parameters to be transferred to GPU
  CurvilinearParameters *startPars;
  GPUERRCHK(cudaMallocHost((void **)&startPars, parsBytes));
  // Copy to the pinned memory
  memcpy(startPars, startParsCollection.data(), parsBytes); 

  // Prepare to perform fit to the created tracks
  KalmanFitterType kFitter(propagator);
  KalmanFitterOptions<VoidOutlierFinder> kfOptions(
      gctx, mctx);
  //KalmanFitterOptions<VoidOutlierFinder> kfOptions(
  //    gctx, mctx, Acts::VoidOutlierFinder(), nullptr, multipleScattering, energyLoss);
  // Pinned mememory for KF fitted tracks
  TSType *fittedTracks;
  GPUERRCHK(cudaMallocHost((void **)&fittedTracks, tsBytes));

  float ms; // elapsed time in milliseconds

  // Create events and streams
  cudaEvent_t startEvent, stopEvent;
  cudaStream_t stream[nStreams];
  GPUERRCHK(cudaEventCreate(&startEvent));
  GPUERRCHK(cudaEventCreate(&stopEvent));
  for (int i = 0; i < nStreams; ++i) {
    GPUERRCHK(cudaStreamCreate(&stream[i]));
  }

  // Running directly on host or offloading to GPU
  bool useGPU = (device == "gpu");
  if (useGPU) {
    GPUERRCHK(cudaEventRecord(startEvent, 0));

    // Allocate memory on device
    PixelSourceLink *d_sourcelinks;
    CurvilinearParameters *d_pars;
    KalmanFitterType *d_kFitter;
    TSType *d_fittedTracks;
    GPUERRCHK(cudaMalloc(&d_sourcelinks, sourcelinksBytes));
    GPUERRCHK(cudaMalloc(&d_pars, parsBytes));
    GPUERRCHK(cudaMalloc(&d_fittedTracks, tsBytes));
    GPUERRCHK(cudaMalloc(&d_kFitter, sizeof(KalmanFitterType)));

    // Copy the KalmanFitter from host to device (shared between all tracks)
    GPUERRCHK(cudaMemcpy(d_kFitter, &kFitter, sizeof(KalmanFitterType),
                         cudaMemcpyHostToDevice));

    // Run on device
    // for (int _ : {1, 2, 3, 4, 5}) {
    for (int i = 0; i < nStreams; ++i) {
      int offset = i * tracksPerStream;
      // Note: need special handling here
      const int streamTracks =
          i == (nStreams - 1) ? tracksLastStream : tracksPerStream;
      // The bytes per stream for source links
      const int sBytes = i == (nStreams - 1) ? lastStreamSourcelinksBytes
                                             : perStreamSourcelinksBytes;
      // The bytes per stream for starting parameters
      const int pBytes =
          i == (nStreams - 1) ? lastStreamParsBytes : perStreamParsBytes;
      // The bytes per stream for fitted tracks
      const int tBytes =
          i == (nStreams - 1) ? lastStreamTSsBytes : perStreamTSsBytes;

      if (i == 0) {
        // @note: prefetch the surface or not
        cudaMemPrefetchAsync(surfaces, surfaceBytes, devId, stream[i]);
      }

      GPUERRCHK(cudaMemcpyAsync(&d_sourcelinks[offset * nSurfaces],
                                &sourcelinks[offset * nSurfaces], sBytes,
                                cudaMemcpyHostToDevice, stream[i]));
      GPUERRCHK(cudaMemcpyAsync(&d_pars[offset], &startPars[offset], pBytes,
                                cudaMemcpyHostToDevice, stream[i]));
      GPUERRCHK(cudaMemcpyAsync(&d_fittedTracks[offset], &fittedTracks[offset],
                                tBytes, cudaMemcpyHostToDevice, stream[i]));

      // Use shared memory for one track if requested
      if (useSharedMemory) {
        fitKernelBlockPerTrack<<<grid, block, 0, stream[i]>>>(
            d_kFitter, d_sourcelinks, d_pars, kfOptions, d_fittedTracks,
            surfacePtrs, nSurfaces, streamTracks, offset);
      } else {
        fitKernelThreadPerTrack<<<grid, block, 0, stream[i]>>>(
            d_kFitter, d_sourcelinks, d_pars, kfOptions, d_fittedTracks,
            surfacePtrs, nSurfaces, streamTracks, offset);
      }
      GPUERRCHK(cudaEventRecord(stopEvent, stream[i]));
      GPUERRCHK(cudaEventSynchronize(stopEvent));
      // copy the fitted tracks to host
      GPUERRCHK(cudaMemcpyAsync(&fittedTracks[offset], &d_fittedTracks[offset],
                                tBytes, cudaMemcpyDeviceToHost, stream[i]));
    }

    GPUERRCHK(cudaPeekAtLastError());
    GPUERRCHK(cudaDeviceSynchronize());

    // Free the memory on device
    GPUERRCHK(cudaFree(d_sourcelinks));
    GPUERRCHK(cudaFree(d_pars));
    GPUERRCHK(cudaFree(d_fittedTracks));
    GPUERRCHK(cudaFree(d_kFitter));

    GPUERRCHK(cudaEventRecord(stopEvent, 0));
    GPUERRCHK(cudaEventSynchronize(stopEvent));
    GPUERRCHK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("Time (ms) for KF memory transfer and execution: %f\n", ms);

    // Log the execution time in seconds (not including the managed memory
    // allocation time for the surfaces)
    Logger::logTime(Logger::buildFilename("nTracks", std::to_string(nTracks),
                                          "gridSize", dim3ToString(grid),
                                          "blockSize", dim3ToString(block)),
                    ms / 1000);

  } else {
    //// Run on host
    auto start_fit = std::chrono::high_resolution_clock::now();

#pragma omp parallel for
    for (int it = 0; it < nTracks; it++) {
      // The fit result wrapper
      KalmanFitterResultType kfResult;
      kfResult.fittedStates =
          Acts::CudaKernelContainer<TSType>(&fittedTracks[it * nSurfaces], nSurfaces);
      // The input source links wrapper
      auto sourcelinkTrack = Acts::CudaKernelContainer<PixelSourceLink>(
          sourcelinks + it * nSurfaces, nSurfaces);
      // Run the fit. The fittedTracks will be changed here
      auto fitStatus = kFitter.fit(sourcelinkTrack, startParsCollection[it],
                                   kfOptions, kfResult, surfacePtrs, nSurfaces);
      if (not fitStatus) {
        std::cout << "fit failure for track " << it << std::endl;
      }
    }
    auto end_fit = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end_fit - start_fit;
    std::cout << "Time (ms) to run KalmanFitter for " << nTracks << " : "
              << elapsed_seconds.count() * 1000 << std::endl;
  }

  if (output) {
    std::cout << "writing KF results" << std::endl;
    Test::writeTracks(fittedTracks, nTracks, nSurfaces);
  }

  std::cout << "------------------------  ending  -----------------------"
            << std::endl;

  // Free the managed/pinned memory
  GPUERRCHK(cudaFree(surfaces));
  GPUERRCHK(cudaFreeHost(sourcelinks));
  GPUERRCHK(cudaFreeHost(startPars));
  GPUERRCHK(cudaFreeHost(fittedTracks));

  return 0;
}
