#include "EigenStepper.hpp"
#include "Propagator.hpp"
#include "TrackParameters.hpp"
#include <fstream>
#include <iostream>
#include <random>
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
__global__ void propKernel(PropagatorType* propagator,
                           PropagatorOptions* propOptions,
                           TrackParameters *tpars, PropResultType* propResult,
                           int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N) {
    propagator->propagate(tpars[i], *propOptions, propResult[i]);
  }
}

int main(int argc, char *argv[]) {
  const int nTracks = 1000;
  std::cout << "----- Propgation test -----" << std::endl;

  // Construct a stepper
  Stepper stepper;
  // Construct a propagator
  PropagatorType propagator(stepper);
  // Construct the propagation options object
  PropagatorOptions propOptions;
  propOptions.maxSteps = 100;

  PropagatorType* d_propagator;
  PropagatorOptions* d_opt;
  
  cudaMalloc((void**)&d_propagator, sizeof(PropagatorType));
  cudaMalloc((void**)&d_opt, sizeof(PropagatorOptions));

  cudaMemcpy(d_propagator, &propagator, sizeof(PropagatorType), cudaMemcpyHostToDevice);
  cudaMemcpy(d_opt, &propOptions, sizeof(PropagatorOptions), cudaMemcpyHostToDevice);

  std::normal_distribution<double> gauss(0., 1.);
  std::default_random_engine generator(42);

  // Construct the starting track parameters
  thrust::device_vector<TrackParameters> parsContainer(nTracks);
  for (int i = 0; i < nTracks; i++) {
    Vector3D rPos(10 * gauss(generator), 10 * gauss(generator), 0);
    Vector3D rMom(25 * gauss(generator), 25 * gauss(generator), 100);
    double q = 1;
    TrackParameters rStart(rPos, rMom, q);
    parsContainer.push_back(rStart);
  }
  TrackParameters *tpars = thrust::raw_pointer_cast(&parsContainer[0]);

  // Propagation result
  thrust::device_vector<PropResultType> resultContainer(nTracks);
  PropResultType *propResult = thrust::raw_pointer_cast(&resultContainer[0]);

  // Run the propagation
  // for (size_t i = 0; i < nTracks; i++) {
  //  propagator.propagate(parsContainer[i], propOptions, resultContainer[i]);
  //}

  int threadsPerBlock = 256;
  int blocksPerGrid = (nTracks + threadsPerBlock - 1) / threadsPerBlock;
  propKernel<<<blocksPerGrid, threadsPerBlock>>>(d_propagator, d_opt, tpars,
                                                 propResult, nTracks);

  /*
  std::cout << "- yielded " << traj.size() << " steps " << std::endl;
  std::cout << "--------------------------" << std::endl;


  std::ofstream obj_track;
  unsigned int iv = 1;
  obj_track.open("Track.obj");
  for (auto &step : traj) {
    obj_track << "v " << step.first.x() << " " << step.first.y() << " "
              << step.first.z() << std::endl;
  }

  for (unsigned int iv = 2; iv <= traj.size(); ++iv) {
    obj_track << "l " << iv - 1 << " " << iv << std::endl;
  }

  obj_track.close();
*/

  return 0;
}
