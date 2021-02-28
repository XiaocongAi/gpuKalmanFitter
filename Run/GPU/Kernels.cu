#pragma once

#include "FitData.hpp"

// Device code
__global__ void
//__launch_bounds__(256, 2)
fitKernelThreadPerTrack(KalmanFitterType *kFitter,
                        Acts::PixelSourceLink *sourcelinks,
                        BoundState *startStates,
                        Acts::LineSurface *targetSurfaces,
                        FitOptionsType *fitOptions, TSType *fittedStates,
                        Acts::BoundParameters<Acts::LineSurface> *fitPars,
                        bool *fitStatus, const PlaneSurfaceType *surfaces,
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
    KalmanFitterResultType fitResult;
    fitResult.fittedStates = Acts::CudaKernelContainer<TSType>(
        fittedStates + threadId * nSurfaces, nSurfaces);
    // Construct a start parameters (the geoContext is set to 0)
    Acts::BoundParameters<Acts::LineSurface> startPars(
        0, startStates[threadId].boundCov, startStates[threadId].boundParams,
        &targetSurfaces[threadId]);
    // Reset the target surface
    fitOptions[threadId].referenceSurface = &targetSurfaces[threadId];
    // Perform the fit
    fitStatus[threadId] = kFitter->fit(
        Acts::CudaKernelContainer<Acts::PixelSourceLink>(
            sourcelinks + threadId * nSurfaces, nSurfaces),
        startPars, fitOptions[threadId], fitResult, surfaces, nSurfaces);
    // Set the fitted parameters
    // @WARNING The reference surface in fPars doesn't make sense actually
    fitPars[threadId] = fitResult.fittedParameters;
  }
}

__global__ void
//__launch_bounds__(256, 2)
fitKernelBlockPerTrack(KalmanFitterType *kFitter,
                       Acts::PixelSourceLink *sourcelinks,
                       BoundState *startStates,
                       Acts::LineSurface *targetSurfaces,
                       FitOptionsType *fitOptions, TSType *fittedStates,
                       Acts::BoundParameters<Acts::LineSurface> *fitPars,
                       bool *fitStatus, const PlaneSurfaceType *surfaces,
                       int nSurfaces, int nTracks, int offset) {
  // @note This will have problem if the number of blocks is smaller than the
  // number of tracks!!!
  int blockId = gridDim.x * blockIdx.y + blockIdx.x + offset;

  // All threads in this block handles the same track
  if (blockId < (nTracks + offset)) {
    // Use the CudaKernelContainer for the source links and fitted states
    // @note shared memory for the fitResult?
    __shared__ KalmanFitterResultType fitResult;
    __shared__ Acts::BoundParameters<Acts::LineSurface> startPars;
    if (threadIdx.x == 0 and threadIdx.y == 0) {
      fitResult = KalmanFitterResultType();
      fitResult.fittedStates = Acts::CudaKernelContainer<TSType>(
          fittedStates + blockId * nSurfaces, nSurfaces);
      // Construct a start parameters (the geoContext is set to 0)
      startPars = Acts::BoundParameters<Acts::LineSurface>(
          0, startStates[blockId].boundCov, startStates[blockId].boundParams,
          &targetSurfaces[blockId]);
      // Reset the target surface
      fitOptions[blockId].referenceSurface = &targetSurfaces[blockId];
    }
    __syncthreads();
    // Perform the fit
    kFitter->fitOnDevice(Acts::CudaKernelContainer<Acts::PixelSourceLink>(
                             sourcelinks + blockId * nSurfaces, nSurfaces),
                         startPars, fitOptions[blockId], fitResult,
                         fitStatus[blockId], surfaces, nSurfaces);
    // Set the fitted parameters with the main thread
    // @WARNING The reference surface in fPars doesn't make sense actually
    if (threadIdx.x == 0 and threadIdx.y == 0) {
      fitPars[blockId] = fitResult.fittedParameters;
      // printf("fittedParams = %f, %f, %f\n", fitPars[blockId].position().x(),
      //       fitPars[blockId].position().y(),
      //       fitPars[blockId].position().z());
    }
    __syncthreads();
  }
}
