#pragma once

#include "FitData.hpp"

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
      // printf("fittedParams = %f, %f, %f\n", fpars[blockId].position().x(),
      //       fpars[blockId].position().y(), fpars[blockId].position().z());
    }
    __syncthreads();
  }
}

