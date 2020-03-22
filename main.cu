#include "EigenStepper.hpp"
#include "Propagator.hpp"
#include "TrackParameters.hpp"
#include <thrust/device_vector.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

static void show_usage(std::string name) {
  std::cerr
      << "Usage: <option(s)> VALUES"
      << "Options:\n"
      << "\t-h,--help\t\tShow this help message\n"
      << "\t-t,--tracks \tSpecify the number of tracks\n"
      << "\t-o,--output \tSpecify the output path for propagation results\n"
      << "\t-d,--device \tSpecify the device: 'gpu' or 'cpu'\n"
      << std::endl;
}

// Struct for B field
struct BField {
  __host__ __device__ static Vector3D getField(const Vector3D & /*field*/) {
    return Vector3D(0., 0., 2.);
  }
};

constexpr unsigned int maxSteps = 1000;

using namespace Acts;
using Stepper = EigenStepper<BField>;
using PropagatorType = Propagator<Stepper>;
using PropResultType = PropagatorResult<maxSteps>;

// Device code
__global__ void propKernel(PropagatorType *propagator, TrackParameters *tpars,
                           PropagatorOptions *propOptions,
                           PropResultType *propResult, int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N) {
    propagator->propagate(tpars[i], *propOptions, propResult[i]);
    // printf("propResult: position = (%f, %f, %f)",
    // propResult[i].position.col(1).x(), propResult[i].position.col(1).y(),
    // propResult[i].position.col(1).z());
  }
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    show_usage(argv[0]);
    return 1;
  }
  std::vector<std::string> sources;
  unsigned int nTracks;
  std::string outputPath = ".";
  std::string device;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "-h") or (arg == "--help")) {
      show_usage(argv[0]);
      return 0;
    } else if (i + 1 < argc) {
      if ((arg == "-t") or (arg == "--tracks")) {
        nTracks = atoi(argv[++i]);
      } else if ((arg == "-o") or (arg == "--output")) {
        outputPath = argv[++i];
      } else if ((arg == "-d") or (arg == "--device")) {
        device = argv[++i];
      } else {
        std::cerr << "Unknown argument." << std::endl;
        return 1;
      }
    }
  }

  std::cout << "----- Propgation test of " << nTracks << " tracks on " << device
            << ". Results are writing to 'path' " << outputPath << " ----- "
            << std::endl;

  // Construct a stepper
  Stepper stepper;
  // Construct a propagator
  PropagatorType propagator(stepper);
  // Construct the propagation options object
  PropagatorOptions propOptions;
  propOptions.maxSteps = 1000;

  // Construct random starting track parameters
  std::normal_distribution<double> gauss(0., 1.);
  std::default_random_engine generator(42);
  TrackParameters pars[nTracks];
  for (int i = 0; i < nTracks; i++) {
    Vector3D rPos(0.1 * gauss(generator), 0.1 * gauss(generator),
                  0); // Units: mm
    Vector3D rMom(0.5 * gauss(generator), 0.5 * gauss(generator),
                  1); // Units: GeV
    double q = 1;
    TrackParameters rStart(rPos, rMom, q);
    pars[i] = rStart;
    // std::cout << " rPos = (" << pars[i].position().x() << ", "
    //           << pars[i].position().y() << ", " << pars[i].position().z()
    //           << ") " << std::endl;
  }

  // Propagation result
  PropResultType ress[nTracks];

  auto start = std::chrono::high_resolution_clock::now();

  // Running directly on host or offloading to GPU
  bool useGPU = (device == "gpu" ? true : false);
  if (useGPU) {
    // Allocate memory on device
    PropagatorType *d_propagator;
    PropagatorOptions *d_opt;
    TrackParameters *d_pars;
    PropResultType *d_ress;

    cudaMalloc((void **)&d_propagator, sizeof(PropagatorType));
    cudaMalloc((void **)&d_opt, sizeof(PropagatorOptions));
    cudaMalloc(&d_pars, nTracks * sizeof(TrackParameters));
    cudaMalloc(&d_ress, nTracks * sizeof(PropResultType));

    // Copy from host to device
    cudaMemcpy(d_propagator, &propagator, sizeof(PropagatorType),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_opt, &propOptions, sizeof(PropagatorOptions),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_pars, pars, nTracks * sizeof(TrackParameters),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_ress, ress, nTracks * sizeof(PropResultType),
               cudaMemcpyHostToDevice);

    // Run on device
    int threadsPerBlock = 256;
    int blocksPerGrid = (nTracks + threadsPerBlock - 1) / threadsPerBlock;
    propKernel<<<blocksPerGrid, threadsPerBlock>>>(d_propagator, d_pars, d_opt,
                                                   d_ress, nTracks);
    // Copy result from device to host
    cudaMemcpy(ress, d_ress, nTracks * sizeof(PropResultType),
               cudaMemcpyDeviceToHost);

    // Free the memory on device
    cudaFree(d_propagator);
    cudaFree(d_opt);
    cudaFree(d_pars);
    cudaFree(d_ress);
  } else {
    // Run on host
    for (int it = 0; it < nTracks; it++) {
      propagator.propagate(pars[it], propOptions, ress[it]);
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "Time (sec) to run propagation tests: "
            << elapsed_seconds.count() << std::endl;

  // Write result to obj file
  std::cout << "Writing yielded " << nTracks << " tracks to obj files..."
            << std::endl;

  for (int it = 0; it < nTracks; it++) {
    PropResultType res = ress[it];
    std::ofstream obj_track;
    std::string fileName = outputPath + "/Track-" + std::to_string(it) + ".obj";
    obj_track.open(fileName.c_str());

    for (int iv = 0; iv < res.steps(); iv++) {
      obj_track << "v " << res.position.col(iv).x() << " "
                << res.position.col(iv).y() << " " << res.position.col(iv).z()
                << std::endl;
    }
    for (unsigned int iv = 2; iv <= res.steps(); ++iv) {
      obj_track << "l " << iv - 1 << " " << iv << std::endl;
    }

    obj_track.close();
  }

  std::cout << "------------------------  ending  -----------------------"
            << std::endl;

  return 0;
}
