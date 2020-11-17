#include "EventData/PixelSourceLink.hpp"
#include "EventData/TrackParameters.hpp"
#include "Fitter/GainMatrixSmoother.hpp"
#include "Fitter/GainMatrixUpdater.hpp"
#include "Fitter/KalmanFitter.hpp"
#include "Material/HomogeneousSurfaceMaterial.hpp"
#include "Propagator/EigenStepper.hpp"
#include "Propagator/Propagator.hpp"
#include "Utilities/CudaHelper.hpp"
#include "Utilities/ParameterDefinitions.hpp"
#include "Utilities/Profiling.hpp"
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
            << "\t-r,--streams \tSpecify number of streams\n"
            // << "\t-p,--pt \tSpecify the pt of particle\n"
            << "\t-o,--output \tIndicator for writing propagation results\n"
            << "\t-d,--device \tSpecify the device: 'gpu' or 'cpu'\n"
            << "\t-g,--grid-size \tSpecify GPU grid size: 'x*y'\n"
            << "\t-b,--block-size \tSpecify GPU block size: 'x*y*z'\n"
            << "\t-s,--shared-memory \tIndicator for using shared memory for "
               "one track or not\n"
            << std::endl;
}

struct BoundState {
  Acts::BoundVector boundParams;
  Acts::BoundMatrix boundCov;
};

using Stepper = Acts::EigenStepper<Test::ConstantBField>;
using PropagatorType = Acts::Propagator<Stepper>;
using PropResultType = Acts::PropagatorResult;
using PropOptionsType = Acts::PropagatorOptions<Simulator, Test::VoidAborter>;
using Smoother = GainMatrixSmoother<Acts::BoundParameters<PlaneSurfaceType>>;
using KalmanFitterType =
    Acts::KalmanFitter<PropagatorType, Acts::GainMatrixUpdater, Smoother>;
using KalmanFitterResultType =
    Acts::KalmanFitterResult<Acts::PixelSourceLink,
                             Acts::BoundParameters<PlaneSurfaceType>,
                             Acts::LineSurface>;
using TSType = typename KalmanFitterResultType::TrackStateType;
using FitOptionsType = Acts::KalmanFitterOptions<Acts::VoidOutlierFinder>;

// Device code
__global__ void __launch_bounds__(256, 2)
    fitKernelThreadPerTrack(KalmanFitterType *kFitter,
                            Acts::PixelSourceLink *sourcelinks,
                            BoundState *bStates, Acts::LineSurface *tarSurfaces,
                            FitOptionsType *kfOptions, TSType *fittedStates,
                            Acts::BoundParameters<Acts::LineSurface> *fpars,
                            bool *fitStatus, const Acts::Surface *surfacePtrs,
                            int nSurfaces, int nTracks, int offset) {
  // In case of 1D grid and 1D block, the threadId = blockDim.x*blockIdx.x +
  // threadIdx.x + offset
  // @note This might have problem if the number of threads is smaller than the
  // number of tracks!!!
  int threadId =
      blockDim.x * blockDim.y * (gridDim.x * blockIdx.y + blockIdx.x) +
      blockDim.x * threadIdx.y + threadIdx.x + offset;

  // Different threads handles different track
  if (threadId < (nTracks + offset)) {
    // Use the CudaKernelContainer for the source links and fitted states
    KalmanFitterResultType kfResult;
    kfResult.fittedStates = CudaKernelContainer<TSType>(
        fittedStates + threadId * nSurfaces, nSurfaces);
    // Construct a start parameters (the geoContext is set to 0)
    Acts::BoundParameters<Acts::LineSurface> startPars(
        0, bStates[threadId].boundCov, bStates[threadId].boundParams,
        &tarSurfaces[threadId]);
    // Reset the target surface
    kfOptions[threadId].referenceSurface = &tarSurfaces[threadId];
    // Perform the fit
    fitStatus[threadId] = kFitter->fit(
        Acts::CudaKernelContainer<PixelSourceLink>(
            sourcelinks + threadId * nSurfaces, nSurfaces),
        startPars, kfOptions[threadId], kfResult, surfacePtrs, nSurfaces);
    // Set the fitted parameters
    // @WARNING The reference surface in fPars doesn't make sense actually
    fpars[threadId] = kfResult.fittedParameters;
  }
}

__global__ void __launch_bounds__(256, 2)
    fitKernelBlockPerTrack(KalmanFitterType *kFitter,
                           Acts::PixelSourceLink *sourcelinks,
                           BoundState *bStates, Acts::LineSurface *tarSurfaces,
                           FitOptionsType *kfOptions, TSType *fittedStates,
                           Acts::BoundParameters<Acts::LineSurface> *fpars,
                           bool *fitStatus, const Acts::Surface *surfacePtrs,
                           int nSurfaces, int nTracks, int offset) {
  // @note This will have problem if the number of blocks is smaller than the
  // number of tracks!!!
  int blockId = gridDim.x * blockIdx.y + blockIdx.x + offset;

  // All threads in this block handles the same track
  if (blockId < (nTracks + offset)) {
    // Use the CudaKernelContainer for the source links and fitted states
    // @note shared memory for the kfResult?
    __shared__ KalmanFitterResultType kfResult;
    __shared__ Acts::BoundParameters<Acts::LineSurface> startPars;
    if (threadIdx.x == 0 and threadIdx.y == 0) {
      kfResult = KalmanFitterResultType();
      kfResult.fittedStates = CudaKernelContainer<TSType>(
          fittedStates + blockId * nSurfaces, nSurfaces);
      // Construct a start parameters (the geoContext is set to 0)
      startPars = Acts::BoundParameters<Acts::LineSurface>(
          0, bStates[blockId].boundCov, bStates[blockId].boundParams,
          &tarSurfaces[blockId]);
      // Reset the target surface
      kfOptions[blockId].referenceSurface = &tarSurfaces[blockId];
    }
    __syncthreads();
    // Perform the fit
    kFitter->fitOnDevice(Acts::CudaKernelContainer<PixelSourceLink>(
                             sourcelinks + blockId * nSurfaces, nSurfaces),
                         startPars, kfOptions[blockId], kfResult,
                         fitStatus[blockId], surfacePtrs, nSurfaces);
    // Set the fitted parameters with the main thread
    // @WARNING The reference surface in fPars doesn't make sense actually
    if (threadIdx.x == 0 and threadIdx.y == 0) {
      fpars[blockId] = kfResult.fittedParameters;
      //printf("fittedParams = %f, %f, %f\n", fpars[blockId].position().x(),
      //       fpars[blockId].position().y(), fpars[blockId].position().z());
    }
    __syncthreads();
  }
}

int main(int argc, char *argv[]) {
  unsigned int nTracks = 10000;
  unsigned int nStreams = 1;
  bool output = false;
  bool useSharedMemory = false;
  std::string device = "cpu";
  std::string bFieldFileName;
  // double p = 1 * Acts::units::_GeV;
  dim3 grid(20000), block(8, 8);
  // This should always be included
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "-h") or (arg == "--help")) {
      show_usage(argv[0]);
      return 0;
    } else if (i + 1 < argc) {
      if ((arg == "-t") or (arg == "--tracks")) {
        nTracks = atoi(argv[++i]);
      } else if ((arg == "-r") or (arg == "--streams")) {
        nStreams = atoi(argv[++i]);
        //} else if ((arg == "-p") or (arg == "--pt")) {
        //  p = atof(argv[++i]) * Acts::units::_GeV;
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

  // The last stream could could less tracks
  const unsigned int tracksPerStream = (nTracks + nStreams - 1) / nStreams;
  const unsigned int tracksLastStream =
      tracksPerStream - (tracksPerStream * nStreams - nTracks);
  std::cout << "tracksPerStream : tracksLastStream = " << tracksPerStream
            << " : " << tracksLastStream << std::endl;

  // @note shall we use this for the grid size?
  const unsigned int blocksPerGrid =
      (tracksPerStream + tracksPerBlock - 1) / tracksPerBlock;
  if (grid.x * grid.y < blocksPerGrid) {
    std::cout << "Grid size too small. It should be at least " << blocksPerGrid
              << std::endl;
    return 1;
  }

  // The shared memory size
  using PropState = PropagatorType::State<PropOptionsType>;
  int sharedMemoryPerTrack = sizeof(PathLimitReached) + sizeof(PropState) +
                             sizeof(bool) * 2 + sizeof(PropagatorResult);
  std::cout << "shared memory is " << sharedMemoryPerTrack << std::endl;

  // The number of navigation surfaces
  const unsigned int nSurfaces = 10;
  // The navigation surfaces is common for all tracks
  const unsigned int navigationSurfaceBytes =
      sizeof(PlaneSurfaceType) * nSurfaces;
  // The track-specific objects
  const unsigned int sourcelinksBytes =
      sizeof(PixelSourceLink) * nSurfaces * nTracks;
  // const unsigned int sParsBytes = sizeof(Acts::CurvilinearParameters) *
  // nTracks;
  const unsigned int bStatesBytes = sizeof(BoundState) * nTracks;
  const unsigned int fParsBytes =
      sizeof(Acts::BoundParameters<Acts::LineSurface>) * nTracks;
  const unsigned int tsBytes = sizeof(TSType) * nSurfaces * nTracks;
  const unsigned int targetSurfaceBytes = sizeof(Acts::LineSurface) * nTracks;
  const unsigned int statusBytes = sizeof(bool) * nTracks;
  const unsigned int optionBytes = sizeof(FitOptionsType) * nTracks;
  std::cout << "surface Bytes = " << navigationSurfaceBytes << std::endl;
  std::cout << "source links Bytes = " << sourcelinksBytes << std::endl;
  // std::cout << "start pars Bytes = " << sParsBytes << std::endl;
  std::cout << "start pars Bytes = " << bStatesBytes << std::endl;
  std::cout << "kf options Bytes = " << optionBytes << std::endl;
  std::cout << "fit states Bytes = " << tsBytes << std::endl;
  std::cout << "fit pars Bytes = " << fParsBytes << std::endl;
  std::cout << "target surface Bytes = " << targetSurfaceBytes << std::endl;

  const unsigned int perSourcelinksBytes =
      sizeof(Acts::PixelSourceLink) * nSurfaces * tracksPerStream;
  const unsigned int lastSourcelinksBytes =
      sizeof(Acts::PixelSourceLink) * nSurfaces * tracksLastStream;

  const unsigned int perTSsBytes = sizeof(TSType) * nSurfaces * tracksPerStream;
  const unsigned int lastTSsBytes =
      sizeof(TSType) * nSurfaces * tracksLastStream;

  // const unsigned int perStartParsBytes =
  //    sizeof(Acts::CurvilinearParameters) * tracksPerStream;
  // const unsigned int lastStartParsBytes =
  //    sizeof(Acts::CurvilinearParameters) * tracksLastStream;
  const unsigned int perStartParsBytes = sizeof(BoundState) * tracksPerStream;
  const unsigned int lastStartParsBytes = sizeof(BoundState) * tracksLastStream;

  const unsigned int perFitParsBytes =
      sizeof(Acts::BoundParameters<Acts::LineSurface>) * tracksPerStream;
  const unsigned int lastFitParsBytes =
      sizeof(Acts::BoundParameters<Acts::LineSurface>) * tracksLastStream;

  const unsigned int perTarSurfacesBytes =
      sizeof(Acts::LineSurface) * tracksPerStream;
  const unsigned int lastTarSurfacesBytes =
      sizeof(Acts::LineSurface) * tracksLastStream;

  const unsigned int perStatusBytes = sizeof(bool) * tracksPerStream;
  const unsigned int lastStatusBytes = sizeof(bool) * tracksLastStream;

  const unsigned int perOptionBytes = sizeof(FitOptionsType) * tracksPerStream;
  const unsigned int lastOptionBytes =
      sizeof(FitOptionsType) * tracksLastStream;

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
  // The silicon material
  Acts::MaterialSlab matProp(Test::makeSilicon(), 0.5 * Acts::units::_mm);
  Acts::HomogeneousSurfaceMaterial surfaceMaterial(matProp);
  // Create plane surfaces without boundaries
  PlaneSurfaceType *surfaces;
  // Unified memory allocation for geometry
  GPUERRCHK(cudaMallocManaged(&surfaces, sizeof(PlaneSurfaceType) * nSurfaces));
  std::cout << "Allocating the memory for the surfaces" << std::endl;
  for (unsigned int isur = 0; isur < nSurfaces; isur++) {
    surfaces[isur] = PlaneSurfaceType(translations[isur],
                                      Acts::Vector3D(1, 0, 0), surfaceMaterial);
    if (not surfaces[isur].surfaceMaterial().materialSlab()) {
      std::cerr << "No surface material" << std::endl;
    }
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
  auto start_propagate = std::chrono::high_resolution_clock::now();
  std::cout << "start to run propagation" << std::endl;
  // Run the simulation to generate sim hits
  // @note We will pick up the valid particles
  std::vector<Simulator::result_type> simResult(nTracks);
  std::vector<ActsFatras::Particle> validParticles(nTracks);
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

  // Build the target surfaces based on the truth particle position
  auto targetSurfacesCollection = buildTargetSurfaces(validParticles);
  // Pinned memory for the target surfaces
  Acts::LineSurface *targetSurfaces;
  GPUERRCHK(cudaMallocHost((void **)&targetSurfaces, targetSurfaceBytes));
  memcpy(targetSurfaces, targetSurfacesCollection.data(), targetSurfaceBytes);

  // The hit smearing resolution
  std::array<double, 2> hitResolution = {30. * Acts::units::_mm,
                                         30. * Acts::units::_mm};
  // Pinned memory for source links
  Acts::PixelSourceLink *sourcelinks;
  GPUERRCHK(cudaMallocHost((void **)&sourcelinks, sourcelinksBytes));
  // Run hit smearing to create source links
  // @note pass the concreate PlaneSurfaceType pointer here
  runHitSmearing(gctx, rng, simResult, hitResolution, sourcelinks, surfaces,
                 nSurfaces);

  // The particle smearing resolution
  ParticleSmearingParameters seedResolution;
  // Run truth seed smearing to create starting parameters with provided
  // reference surface
  auto startParsCollection = runParticleSmearing(
      rng, gctx, validParticles, seedResolution, targetSurfaces, nTracks);
  // Pinned memory for starting track parameters to be transferred to GPU
  BoundState *boundStates;
  GPUERRCHK(cudaMallocHost((void **)&boundStates, bStatesBytes));
  // Initialize the boundState
  for (unsigned int it = 0; it < nTracks; it++) {
    boundStates[it].boundParams = startParsCollection[it].parameters();
    boundStates[it].boundCov = *startParsCollection[it].covariance();
  }

  // The fitted parameters
  Acts::BoundParameters<Acts::LineSurface> *fitPars;
  GPUERRCHK(cudaMallocHost((void **)&fitPars, fParsBytes));

  // Prepare to perform fit to the created tracks
  KalmanFitterType kFitter(propagator);

  // Pinned memory for KF options
  FitOptionsType *kfOptions;
  GPUERRCHK(cudaMallocHost((void **)&kfOptions, optionBytes));
  // Pinned memory for KF fitted tracks
  TSType *fittedStates;
  GPUERRCHK(cudaMallocHost((void **)&fittedStates, tsBytes));
  // Pinned memory for KF fitted parameters
  // Pinned memory for KF fit status
  bool *fitStatus;
  GPUERRCHK(cudaMallocHost((void **)&fitStatus, statusBytes));
  // Initialize the kfOptions and fit status
  for (unsigned int it = 0; it < nTracks; it++) {
    kfOptions[it] = FitOptionsType(gctx, mctx);
    // kfOptions[it].referenceSurface = &startPars[it].referenceSurface();
    fitStatus[it] = false;
  }

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
    KalmanFitterType *d_kFitter;
    Acts::PixelSourceLink *d_sourcelinks;
    // The start pars will be constructed on GPU with bound vector and
    // covariance
    // Acts::BoundParameters<Acts::LineSurface> *d_startPars;
    BoundState *d_boundStates;
    Acts::BoundParameters<Acts::LineSurface> *d_fitPars;
    FitOptionsType *d_kfOptions;
    TSType *d_fittedStates;
    bool *d_fitStatus;
    Acts::LineSurface *d_targetSurfaces;

    GPUERRCHK(cudaMalloc(&d_kFitter, sizeof(KalmanFitterType)));
    GPUERRCHK(cudaMalloc(&d_sourcelinks, sourcelinksBytes));
    // GPUERRCHK(cudaMalloc(&d_startPars, sParsBytes));
    GPUERRCHK(cudaMalloc(&d_boundStates, bStatesBytes));
    GPUERRCHK(cudaMalloc(&d_fitPars, fParsBytes));
    GPUERRCHK(cudaMalloc(&d_kfOptions, optionBytes));
    GPUERRCHK(cudaMalloc(&d_fittedStates, tsBytes));
    GPUERRCHK(cudaMalloc(&d_fitStatus, statusBytes));
    GPUERRCHK(cudaMalloc(&d_targetSurfaces, targetSurfaceBytes));

    // Copy the KalmanFitter from host to device (shared between all tracks)
    GPUERRCHK(cudaMemcpy(d_kFitter, &kFitter, sizeof(KalmanFitterType),
                         cudaMemcpyHostToDevice));

    // Run on device
    for (unsigned int i = 0; i < nStreams; ++i) {
      unsigned int offset = i * tracksPerStream;
      // The number of tracks handled in this stream
      unsigned int streamTracks = tracksPerStream;
      unsigned int slBytes = perSourcelinksBytes;
      unsigned int spBytes = perStartParsBytes;
      unsigned int fpBytes = perFitParsBytes;
      unsigned int tsBytes = perTSsBytes;
      unsigned int stBytes = perStatusBytes;
      unsigned int opBytes = perOptionBytes;
      unsigned int asBytes = perTarSurfacesBytes;
      if (i == (nStreams - 1)) {
        streamTracks = tracksLastStream;
        slBytes = lastSourcelinksBytes;
        spBytes = lastStartParsBytes;
        fpBytes = lastFitParsBytes;
        tsBytes = lastTSsBytes;
        stBytes = lastStatusBytes;
        opBytes = lastOptionBytes;
        asBytes = lastTarSurfacesBytes;
      }

      if (i == 0) {
        // @note: prefetch the surface or not
        cudaMemPrefetchAsync(surfaces, navigationSurfaceBytes, devId,
                             stream[i]);
      }

      // Copy the sourcelinsk, starting parameters and fitted tracks from host
      // to device
      GPUERRCHK(cudaMemcpyAsync(&d_sourcelinks[offset * nSurfaces],
                                &sourcelinks[offset * nSurfaces], slBytes,
                                cudaMemcpyHostToDevice, stream[i]));
      // GPUERRCHK(cudaMemcpyAsync(&d_startPars[offset], &startPars[offset],
      //                           spBytes, cudaMemcpyHostToDevice, stream[i]));
      GPUERRCHK(cudaMemcpyAsync(&d_boundStates[offset], &boundStates[offset],
                                spBytes, cudaMemcpyHostToDevice, stream[i]));
      GPUERRCHK(cudaMemcpyAsync(&d_fitPars[offset], &fitPars[offset], fpBytes,
                                cudaMemcpyHostToDevice, stream[i]));
      GPUERRCHK(cudaMemcpyAsync(&d_fittedStates[offset * nSurfaces],
                                &fittedStates[offset * nSurfaces], tsBytes,
                                cudaMemcpyHostToDevice, stream[i]));
      GPUERRCHK(cudaMemcpyAsync(&d_kfOptions[offset], &kfOptions[offset],
                                opBytes, cudaMemcpyHostToDevice, stream[i]));
      GPUERRCHK(cudaMemcpyAsync(&d_fitStatus[offset], &fitStatus[offset],
                                stBytes, cudaMemcpyHostToDevice, stream[i]));
      GPUERRCHK(cudaMemcpyAsync(&d_targetSurfaces[offset],
                                &targetSurfaces[offset], asBytes,
                                cudaMemcpyHostToDevice, stream[i]));

      // Use shared memory for one track if requested
      if (useSharedMemory) {
        fitKernelBlockPerTrack<<<grid, block, 0, stream[i]>>>(
            d_kFitter, d_sourcelinks, d_boundStates, d_targetSurfaces,
            d_kfOptions, d_fittedStates, d_fitPars, d_fitStatus, surfacePtrs,
            nSurfaces, streamTracks, offset);
      } else {
        fitKernelThreadPerTrack<<<grid, block, 0, stream[i]>>>(
            d_kFitter, d_sourcelinks, d_boundStates, d_targetSurfaces,
            d_kfOptions, d_fittedStates, d_fitPars, d_fitStatus, surfacePtrs,
            nSurfaces, streamTracks, offset);
      }
      GPUERRCHK(cudaEventRecord(stopEvent, stream[i]));
      GPUERRCHK(cudaEventSynchronize(stopEvent));
      // copy the fitted states to host
      GPUERRCHK(cudaMemcpyAsync(&fittedStates[offset * nSurfaces],
                                &d_fittedStates[offset * nSurfaces], tsBytes,
                                cudaMemcpyDeviceToHost, stream[i]));
      // copy the fitted params to host
      GPUERRCHK(cudaMemcpyAsync(&fitPars[offset], &d_fitPars[offset], fpBytes,
                                cudaMemcpyDeviceToHost, stream[i]));
      // copy the fit status to host
      GPUERRCHK(cudaMemcpyAsync(&fitStatus[offset], &d_fitStatus[offset],
                                stBytes, cudaMemcpyDeviceToHost, stream[i]));
    }

    GPUERRCHK(cudaPeekAtLastError());
    GPUERRCHK(cudaDeviceSynchronize());

    // Free the memory on device
    GPUERRCHK(cudaFree(d_sourcelinks));
    // GPUERRCHK(cudaFree(d_startPars));
    GPUERRCHK(cudaFree(d_boundStates));
    GPUERRCHK(cudaFree(d_fittedStates));
    GPUERRCHK(cudaFree(d_fitPars));
    GPUERRCHK(cudaFree(d_kfOptions));
    GPUERRCHK(cudaFree(d_fitStatus));
    GPUERRCHK(cudaFree(d_kFitter));
    GPUERRCHK(cudaFree(d_targetSurfaces));

    GPUERRCHK(cudaEventRecord(stopEvent, 0));
    GPUERRCHK(cudaEventSynchronize(stopEvent));
    GPUERRCHK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("Time (ms) for KF memory transfer and execution: %f\n", ms);

    // Log the execution time in seconds (not including the managed memory
    // allocation time for the surfaces)
    Test::Logger::logTime(
        Test::Logger::buildFilename(
            "timing_gpu", "nTracks", std::to_string(nTracks), "nStreams",
            std::to_string(nStreams), "gridSize", dim3ToString(grid),
            "blockSize", dim3ToString(block), "sharedMemory",
            std::to_string(static_cast<unsigned int>(useSharedMemory))),
        ms / 1000);

  } else {
    /// Run on host
    auto start_fit = std::chrono::high_resolution_clock::now();
#pragma omp parallel for num_threads(250)
    for (int it = 0; it < nTracks; it++) {
      // The fit result wrapper
      KalmanFitterResultType kfResult;
      kfResult.fittedStates = Acts::CudaKernelContainer<TSType>(
          &fittedStates[it * nSurfaces], nSurfaces);
      // @note when it >=35, we got different startPars[i] between CPU and GPU
      // The input source links wrapper
      auto sourcelinkTrack = Acts::CudaKernelContainer<PixelSourceLink>(
          sourcelinks + it * nSurfaces, nSurfaces);
      kfOptions[it].referenceSurface = &targetSurfaces[it];
      // Run the fit. The fittedStates will be changed here
      auto status =
          kFitter.fit(sourcelinkTrack, startParsCollection[it], kfOptions[it],
                      kfResult, surfacePtrs, nSurfaces);
      if (not status) {
        std::cout << "fit failure for track " << it << std::endl;
      }
      fitStatus[it] = status;
      fitPars[it] = kfResult.fittedParameters;
    }
    auto end_fit = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end_fit - start_fit;
    std::cout << "Time (ms) to run KalmanFitter for " << nTracks << " : "
              << elapsed_seconds.count() * 1000 << std::endl;
  }
  int threads = omp_get_num_threads();

  if (output) {
    std::cout << "writing KF results" << std::endl;
    std::string stateFileName;
    std::string paramFileName;
    std::string rootFileName;

    std::string param = "smoothed";
    stateFileName.append("fitted_");
    stateFileName.append(param);

    paramFileName.append("fitted_");
    paramFileName.append("param");

    rootFileName.append("fitted_");
    rootFileName.append("param");
    if (useGPU) {
      stateFileName.append("_gpu_nTracks_");
      paramFileName.append("_gpu_nTracks_");
      rootFileName.append("_gpu_nTracks_");
    } else {
      stateFileName.append("_semi_cpu_nTracks_");
      paramFileName.append("_semi_cpu_nTracks_");
      rootFileName.append("_semi_cpu_nTracks_");
    }
    stateFileName.append(std::to_string(nTracks)).append(".obj");
    paramFileName.append(std::to_string(nTracks)).append(".csv");
    rootFileName.append(std::to_string(nTracks)).append(".root");
    writeStatesObj(fittedStates, fitStatus, nTracks, nSurfaces, stateFileName,
                   param);
    writeParamsCsv(fitPars, fitStatus, nTracks, paramFileName);
    writeParamsRoot(gctx, fitPars, fitStatus, validParticles, nTracks,
                    rootFileName, "params");
  }

  std::cout << "------------------------  ending  -----------------------"
            << std::endl;

  // Free the managed/pinned memory
  GPUERRCHK(cudaFree(surfaces));
  GPUERRCHK(cudaFreeHost(sourcelinks));
  //  GPUERRCHK(cudaFreeHost(startPars));
  GPUERRCHK(cudaFreeHost(boundStates));
  GPUERRCHK(cudaFreeHost(targetSurfaces));
  GPUERRCHK(cudaFreeHost(fittedStates));
  GPUERRCHK(cudaFreeHost(fitStatus));

  return 0;
}
