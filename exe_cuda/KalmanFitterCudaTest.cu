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
#include "Test/TestHelper.hpp"
#include "Utilities/CudaHelper.hpp"
#include "Utilities/ParameterDefinitions.hpp"
#include "Utilities/Units.hpp"

#include "Utilities/Profiling.hpp"

#include "Utilities/Logger.hpp"

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
            << "\t-s,--shared-memory \tIndicator for using shared memory for one track or not\n"
            << std::endl;
}

using namespace Acts;

using Stepper = EigenStepper<ConstantBField>;
// using Stepper = EigenStepper<InterpolatedBFieldMap3D>;
using PropagatorType = Propagator<Stepper>;
using PropResultType = PropagatorResult;
using SimPropOptionsType = PropagatorOptions<MeasurementCreator, VoidAborter>;
using PropState = PropagatorType::State<SimPropOptionsType>;
using KalmanFitterType = KalmanFitter<PropagatorType, GainMatrixUpdater>;
using KalmanFitterResultType =
    KalmanFitterResult<PixelSourceLink, BoundParameters>;
using TSType = typename KalmanFitterResultType::TrackStateType;

using PlaneSurfaceType = PlaneSurface<InfiniteBounds>;

// Device code
__global__ void __launch_bounds__(256, 2)
    fitKernelThreadPerTrack(KalmanFitterType *kFitter, PixelSourceLink *sourcelinks,
              CurvilinearParameters *tpars,
              KalmanFitterOptions<VoidOutlierFinder> kfOptions,
              TSType *fittedTracks, const Surface *surfacePtrs, int nSurfaces,
              int nTracks, int offset) {
  int i = blockDim.x * blockIdx.x + threadIdx.x + offset;
  if (i < (nTracks + offset)) {
    // Use the CudaKernelContainer for the source links and fitted tracks
    KalmanFitterResultType kfResult;
    kfResult.fittedStates =
        CudaKernelContainer<TSType>(fittedTracks + i * nSurfaces, nSurfaces);
    kFitter->fit(CudaKernelContainer<PixelSourceLink>(
                     sourcelinks + i * nSurfaces, nSurfaces),
                 tpars[i], kfOptions, kfResult, surfacePtrs, nSurfaces);
  }
}

__global__ void __launch_bounds__(256, 2)
    fitKernelBlockPerTrack(KalmanFitterType *kFitter, PixelSourceLink *sourcelinks,
              CurvilinearParameters *tpars,
              KalmanFitterOptions<VoidOutlierFinder> kfOptions,
              TSType *fittedTracks, const Surface *surfacePtrs, int nSurfaces,
              int nTracks, int offset) {
  int i = blockDim.x * blockIdx.x + threadIdx.x + offset;
  if (i < (nTracks + offset)) {
    // Use the CudaKernelContainer for the source links and fitted tracks
    KalmanFitterResultType kfResult;
    kfResult.fittedStates =
        CudaKernelContainer<TSType>(fittedTracks + i * nSurfaces, nSurfaces);
    kFitter->fitOnDevice(CudaKernelContainer<PixelSourceLink>(
                     sourcelinks + i * nSurfaces, nSurfaces),
                 tpars[i], kfOptions, kfResult, surfacePtrs, nSurfaces);
  }
}

int main(int argc, char *argv[]) {
  //  if (argc < 5) {
  //    show_usage(argv[0]);
  //    return 1;
  //  }
  unsigned int nTracks = 10240;
  bool output = false;
  bool useSharedMemory = true;
  std::string device = "cpu";
  std::string bFieldFileName;
  double p = 1 * Acts::units::_GeV;
  dim3 grid(40), block(8,8); 
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
      } else if ((arg == "-b") or (arg == "--block-size")){
        block = stringToDim3(argv[++i]);
      } else if ((arg == "-s") or (arg == "--shared-memory")) {
        useSharedMemory = (atoi(argv[++i]) == 1);
      } else {
        std::cerr << "Unknown argument." << std::endl;
        return 1;
      }
    }
  }
  
  std::cout << grid.x << " " << grid.y << " "<< block.x << " " << block.y << std::endl; 

  int devId = 0;

  cudaDeviceProp prop;
  GPUERRCHK(cudaGetDeviceProperties(&prop, devId));
  printf("Device : %s\n", prop.name);
  GPUERRCHK(cudaSetDevice(devId));

  int driverVersion;
  GPUERRCHK(cudaDriverGetVersion(&driverVersion));
  printf("cuda driver version: %i\n", driverVersion);

  int rtVersion;
  GPUERRCHK(cudaRuntimeGetVersion(&rtVersion));
  printf("cuda rt version: %i\n", rtVersion);
 
  // The block dimension
  const int threadsPerBlock = block.x*block.y*block.z;

  const int nStreams = 4;
  // The last stream could could less threads
  const int threadsPerStream = (nTracks + nStreams - 1) / nStreams;
  const int overflowThreads = threadsPerStream * nStreams - nTracks;
  const int threadsLastStream = threadsPerStream - overflowThreads;
  std::cout << "threadsPerStream = " << threadsPerStream << std::endl;
  std::cout << "threadsLastStream = " << threadsLastStream << std::endl;
  
  const int blocksPerGrid_multiStream =
      (threadsPerStream + threadsPerBlock - 1) / threadsPerBlock;

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
  const int perStreamSourcelinksBytes =
      sizeof(PixelSourceLink) * nSurfaces * threadsPerStream;
  const int perStreamParsBytes =
      sizeof(CurvilinearParameters) * threadsPerStream;
  const int perStreamTSsBytes = sizeof(TSType) * nSurfaces * threadsPerStream;

  std::cout << "surface Bytes = " << surfaceBytes << std::endl;
  std::cout << "source links Bytes = " << sourcelinksBytes << std::endl;
  std::cout << "startPars Bytes = " << parsBytes << std::endl;
  std::cout << "TSs Bytes = " << tsBytes << std::endl;

  const int lastStreamSourcelinksBytes =
      sizeof(PixelSourceLink) * nSurfaces * threadsLastStream;
  const int lastStreamParsBytes =
      sizeof(CurvilinearParameters) * threadsLastStream;
  const int lastStreamTSsBytes = sizeof(TSType) * nSurfaces * threadsLastStream;

  // Create a test context
  GeometryContext gctx(0);
  MagneticFieldContext mctx(0);

  // Create the geometry
  // Set translation vectors
  std::vector<Acts::Vector3D> translations;
  for (unsigned int isur = 0; isur < nSurfaces; isur++) {
    translations.push_back({(isur * 30. + 19) * Acts::units::_mm, 0., 0.});
  }

  PlaneSurfaceType *surfaces;
  // Unifited memory allocation for geometry
  GPUERRCHK(cudaMallocManaged(&surfaces, sizeof(PlaneSurfaceType) * nSurfaces));
  std::cout << "Allocating the memory for the surfaces" << std::endl;
  for (unsigned int isur = 0; isur < nSurfaces; isur++) {
    surfaces[isur] =
        PlaneSurfaceType(translations[isur], Acts::Vector3D(1, 0, 0));
  }
  std::cout << "Creating " << nSurfaces << " boundless plane surfaces"
            << std::endl;

  // Test the pointers to surfaces
  for (unsigned int isur = 0; isur < nSurfaces; isur++) {
    auto surface = surfaces[isur];
    std::cout << "surface " << isur << " has center at: \n"
              << surface.center(gctx) << std::endl;
  }

  std::cout << "----- Starting Kalman fitter test of " << nTracks
            << " tracks on " << device << std::endl;

  const Acts::Surface *surfacePtrs = surfaces;

  // InterpolatedBFieldMap3D bField = Options::readBField(bFieldFileName);

  // Construct a stepper with the bField
  Stepper stepper;
  PropagatorType propagator(stepper);
  SimPropOptionsType propOptions(gctx, mctx);
  propOptions.maxSteps = 100;
  propOptions.initializer.surfaceSequence = surfacePtrs;
  propOptions.initializer.surfaceSequenceSize = nSurfaces;

  // Construct random starting track parameters
  CurvilinearParameters *startPars;
  GPUERRCHK(cudaMallocHost((void **)&startPars, parsBytes)); // use pinned
                                                             // memory
  double resLoc1 = 0.1 * Acts::units::_mm;
  double resLoc2 = 0.1 * Acts::units::_mm;
  double resPhi = 0.01;
  double resTheta = 0.01;

  const BoundSymMatrix cov = [=]() {
    BoundSymMatrix cov = BoundSymMatrix::Zero();
    cov << resLoc1 * resLoc1, 0., 0., 0., 0., 0., 0., resLoc2 * resLoc2, 0., 0.,
        0., 0., 0., 0., resPhi * resPhi, 0., 0., 0., 0., 0., 0.,
        resTheta * resTheta, 0., 0., 0., 0., 0., 0., 0.0001, 0., 0., 0., 0., 0.,
        0., 1.;
    return cov;
  }();

  for (int i = 0; i < nTracks; i++) {

    double q = 1;
    double time = 0;
    double phi = gauss(generator) * resPhi;
    double theta = M_PI / 2 + gauss(generator) * resTheta;
    Vector3D pos(0, resLoc1 * gauss(generator),
                 resLoc2 * gauss(generator)); // Units: mm
    Vector3D mom(p * sin(theta) * cos(phi), p * sin(theta) * sin(phi),
                 p * cos(theta)); // Units: GeV

    startPars[i] = CurvilinearParameters(cov, pos, mom, q, time);
  }
  std::cout << "Finish creating starting parameters" << std::endl;

  // Propagation result
  std::vector<MeasurementCreator::result_type> ress(nTracks);

  std::cout << "Start to run propagation to create measurements" << std::endl;
  auto start_propagate = std::chrono::high_resolution_clock::now();

// Run propagation to create the measurements
#pragma omp parallel for
  for (int it = 0; it < nTracks; it++) {
    propagator.propagate(startPars[it], propOptions, ress[it]);
  }

  auto end_propagate = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds =
      end_propagate - start_propagate;
  std::cout << "Time (ms) to run propagation tests: "
            << elapsed_seconds.count() * 1000 << std::endl;

  // Initialize the vertex counter
  unsigned int vCounter = 0;
  if (output) {
    std::cout << "writing propagation results" << std::endl;
    // Write all of the created tracks to one obj file
    std::ofstream obj_track;
    std::string fileName = "Tracks-propagation.obj";
    obj_track.open(fileName.c_str());

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

  // Prepare to perform fit to the created tracks
  float ms; // elapsed time in milliseconds

  // Create events and streams
  cudaEvent_t startEvent, stopEvent;
  cudaStream_t stream[nStreams];
  GPUERRCHK(cudaEventCreate(&startEvent));
  GPUERRCHK(cudaEventCreate(&stopEvent));
  for (int i = 0; i < nStreams; ++i) {
    GPUERRCHK(cudaStreamCreate(&stream[i]));
  }

  // Restore the source links
  PixelSourceLink *sourcelinks;
  GPUERRCHK(cudaMallocHost((void **)&sourcelinks,
                           sourcelinksBytes)); // use pinned memory
  for (int it = 0; it < nTracks; it++) {
    const auto &sls = ress[it].sourcelinks;
    for (int is = 0; is < nSurfaces; is++) {
      sourcelinks[it * nSurfaces + is] = sls[is];
    }
  }

  // Create an KFitter
  PropagatorType rPropagator(stepper);
  KalmanFitterType kFitter(rPropagator);

  // The KF options
  KalmanFitterOptions<VoidOutlierFinder> kfOptions(gctx, mctx);

  // Allocate memory for KF fitted tracks
  TSType *fittedTracks;
  GPUERRCHK(cudaMallocHost((void **)&fittedTracks, tsBytes));

  // Running directly on host or offloading to GPU
  bool useGPU = (device == "gpu" ? true : false);
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

    // Copy the KalmanFitter from host to device (shared between all threads)
    GPUERRCHK(cudaMemcpy(d_kFitter, &kFitter, sizeof(KalmanFitterType),
                         cudaMemcpyHostToDevice));

    // Run on device
    //    for (int _ : {1, 2, 3, 4, 5}) {
    for (int i = 0; i < nStreams; ++i) {
      int offset = i * threadsPerStream;
      // Note: need special handling here
      const int threads =
          (i == (nStreams - 1) ? threadsLastStream : threadsPerStream);
      // The bytes per stream for source links
      const int sBytes = (i == (nStreams - 1) ? lastStreamSourcelinksBytes
                                              : perStreamSourcelinksBytes);
      // The bytes per stream for starting parameters
      const int pBytes =
          (i == (nStreams - 1) ? lastStreamParsBytes : perStreamParsBytes);
      // The bytes per stream for fitted tracks
      const int tBytes =
          (i == (nStreams - 1) ? lastStreamTSsBytes : perStreamTSsBytes);
    
      if(i==0){
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
     if(useSharedMemory){ 
         fitKernelBlockPerTrack<<<grid, block, 0, stream[i]>>>(
          d_kFitter, d_sourcelinks, d_pars, kfOptions, d_fittedTracks,
          surfacePtrs, nSurfaces, threads, offset);
      }else{
        fitKernelThreadPerTrack<<<grid, block, 0,
        stream[i]>>>(
           d_kFitter, d_sourcelinks, d_pars, kfOptions, d_fittedTracks,
           surfacePtrs, nSurfaces, threads, offset);
      }
      GPUERRCHK(cudaEventRecord(stopEvent, stream[i]));
      GPUERRCHK(cudaEventSynchronize(stopEvent));
      // copy the fitted tracks to host
      GPUERRCHK(cudaMemcpyAsync(&fittedTracks[offset], &d_fittedTracks[offset],
                                tBytes, cudaMemcpyDeviceToHost, stream[i]));
    }
    //    }

    GPUERRCHK(cudaPeekAtLastError());
    GPUERRCHK(cudaDeviceSynchronize());

    // Free the memory on device
    GPUERRCHK(cudaFree(d_sourcelinks));
    GPUERRCHK(cudaFree(d_pars));
    GPUERRCHK(cudaFree(d_fittedTracks));
    GPUERRCHK(cudaFree(d_kFitter));
    GPUERRCHK(cudaFree(surfaces));
    GPUERRCHK(cudaFreeHost(sourcelinks));
    GPUERRCHK(cudaFreeHost(startPars));

    GPUERRCHK(cudaEventRecord(stopEvent, 0));
    GPUERRCHK(cudaEventSynchronize(stopEvent));
    GPUERRCHK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("Time (ms) for KF memory transfer and execution: %f\n", ms);

    // Log the execution time in seconds
    Logger::logTime(Logger::buildFilename("nTracks", std::to_string(nTracks), "gridSize", dim3ToString(grid), "blockSize", dim3ToString(block)),
             ms/1000);
 
  } else {
    //// Run on host
    auto start_fit = std::chrono::high_resolution_clock::now();

#pragma omp parallel for
    for (int it = 0; it < nTracks; it++) {
      //     BoundSymMatrix cov = BoundSymMatrix::Zero();
      //     cov << resLoc1 * resLoc1, 0., 0., 0., 0., 0., 0., resLoc2 *
      //     resLoc2, 0.,
      //         0., 0., 0., 0., 0., resPhi * resPhi, 0., 0., 0., 0., 0., 0.,
      //         resTheta * resTheta, 0., 0., 0., 0., 0., 0., 0.0001, 0., 0.,
      //         0., 0., 0., 0., 1.;

      //     double q = 1;
      //     double time = 0;
      //     Vector3D pos(0, 0, 0); // Units: mm
      //     Vector3D mom(p, 0, 0); // Units: GeV

      //     CurvilinearParameters rStart(cov, pos, mom, q, time);

      // Dynamically allocating memory for the fitted states here
      KalmanFitterResultType kfResult;
      kfResult.fittedStates =
          CudaKernelContainer<TSType>(&fittedTracks[it * nSurfaces], nSurfaces);

      auto sourcelinkTrack = CudaKernelContainer<PixelSourceLink>(
          ress[it].sourcelinks.data(), ress[it].sourcelinks.size());

      // The fittedTracks will be changed here
      // Note that we are using exacty the truth starting parameters here (which
      // should be added smearing)
      auto fitStatus = kFitter.fit(sourcelinkTrack, startPars[it], kfOptions,
                                   kfResult, surfacePtrs, nSurfaces);
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
    // Write all of the created tracks to one obj file
    std::ofstream obj_ftrack;
    std::string fileName_ = "Tracks-fitted.obj";
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

  GPUERRCHK(cudaFreeHost(fittedTracks));

  return 0;
}
