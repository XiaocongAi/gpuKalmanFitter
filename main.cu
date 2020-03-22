#include "EigenStepper.hpp"
#include "Propagator.hpp"
#include "TrackParameters.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <thrust/device_vector.h>
#include <vector>

struct BField {
  __host__ __device__ static Vector3D getField(const Vector3D & /*field*/) {
    return Vector3D(0., 0., 2.);
  }
};

using namespace Acts;
using Stepper = EigenStepper<BField>;
using PropagatorType = Propagator<Stepper>;
using PropResultType = PropagatorResult<100>;

// Device code
__global__ void propKernel(PropagatorType *propagator,
                           PropagatorOptions *propOptions,
                           TrackParameters *tpars, PropResultType *propResult,
                           int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N) {
    propagator->propagate(tpars[i], *propOptions, propResult[i]);
    // printf("propResult[i] = %f", propResult[i].position.col(1).x());
  }
}

int main(int argc, char *argv[]) {
  const int nTracks = atoi(argv[1]);
  const int runOnGPU = atoi(argv[2]);
  std::string device = (runOnGPU == 1 ? "GPU" : "CPU");
  std::cout << "----- Propgation test of " << nTracks << " tracks on " << device
            << " ----- " << std::endl;

  // Construct a stepper
  Stepper stepper;
  // Construct a propagator
  PropagatorType propagator(stepper);
  // Construct the propagation options object
  PropagatorOptions propOptions;
  propOptions.maxSteps = 100;

  // Allocate memory on device for propagator and propagator options
  PropagatorType *d_propagator;
  PropagatorOptions *d_opt;
  cudaMalloc((void **)&d_propagator, sizeof(PropagatorType));
  cudaMalloc((void **)&d_opt, sizeof(PropagatorOptions));
  cudaMemcpy(d_propagator, &propagator, sizeof(PropagatorType),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_opt, &propOptions, sizeof(PropagatorOptions),
             cudaMemcpyHostToDevice);

  // Construct the starting track parameters
  // and allocate memory on device
  std::normal_distribution<double> gauss(0., 1.);
  std::default_random_engine generator(42);
  TrackParameters pars[nTracks], *d_pars;
  for (int i = 0; i < nTracks; i++) {
    Vector3D rPos(10 * gauss(generator), 10 * gauss(generator), 0);
    Vector3D rMom(25 * gauss(generator), 25 * gauss(generator), 100);
    double q = 1;
    TrackParameters rStart(rPos, rMom, q);
    pars[i] = rStart;
    std::cout << " rPos = (" << pars[i].position().x() << ", "
              << pars[i].position().y() << ", " << pars[i].position().z()
              << ") " << std::endl;
  }
  cudaMalloc(&d_pars, nTracks * sizeof(TrackParameters));
  cudaMemcpy(d_pars, pars, nTracks * sizeof(TrackParameters),
             cudaMemcpyHostToDevice);

  // Allocate memory for propagation result
  PropResultType ress[nTracks], *d_ress;
  cudaMalloc(&d_ress, nTracks * sizeof(PropResultType));
  cudaMemcpy(d_ress, ress, nTracks * sizeof(PropResultType),
             cudaMemcpyHostToDevice);

  if (runOnGPU == 1) {
    // Run on device
    int threadsPerBlock = 256;
    int blocksPerGrid = (nTracks + threadsPerBlock - 1) / threadsPerBlock;
    propKernel<<<blocksPerGrid, threadsPerBlock>>>(d_propagator, d_opt, d_pars,
                                                   d_ress, nTracks);
    // Copy result from device to host
    cudaMemcpy(ress, d_ress, nTracks * sizeof(PropResultType),
               cudaMemcpyDeviceToHost);
  } else {
    // Run on host
    for (int it = 0; it < nTracks; it++) {
      propagator.propagate(pars[it], propOptions, ress[it]);
    }
  }

  // Write result to obj file
  std::cout << "- yielded " << nTracks << " tracks " << std::endl;
  std::cout << "--------------------------" << std::endl;

  for (int it = 0; it < nTracks; it++) {
    PropResultType res = ress[it];
    std::ofstream obj_track;
    std::string fileName = "Track-" + std::to_string(it) + ".obj";
    obj_track.open(fileName.c_str());

    for (int iv = 0; iv < res.nSteps(); iv++) {
      obj_track << "v " << res.position.col(iv).x() << " "
                << res.position.col(iv).y() << " " << res.position.col(iv).z()
                << std::endl;
    }
    for (unsigned int iv = 2; iv <= res.nSteps(); ++iv) {
      obj_track << "l " << iv - 1 << " " << iv << std::endl;
    }

    obj_track.close();
  }

  cudaFree(d_propagator);
  cudaFree(d_opt);
  cudaFree(d_pars);
  cudaFree(d_ress);

  return 0;
}
