#include "Propagator/EigenStepper.hpp"
#include "Propagator/Propagator.hpp"
#include "EventData/TrackParameters.hpp"
#include "Plugins/BFieldOptions.hpp"
#include "Plugins/BFieldUtils.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define GPUERRCHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}

static void show_usage(std::string name) {
  std::cerr << "Usage: <option(s)> VALUES"
            << "Options:\n"
            << "\t-h,--help\t\tShow this help message\n"
            << "\t-t,--tracks \tSpecify the number of tracks\n"
            << "\t-p,--pt \tSpecify the pt of particle\n"
            << "\t-o,--output \tIndicator for writing propagation results\n"
            << "\t-d,--device \tSpecify the device: 'gpu' or 'cpu'\n"
            << "\t-b,--bf-map \tSpecify the path of *.txt for interpolated BField map\n"
            << std::endl;
}

// Struct for B field
struct ConstantBField {
  ACTS_DEVICE_FUNC static Vector3D getField(const Vector3D & /*field*/) {
    return Vector3D(0., 0., 2.);
  }
};

constexpr unsigned int maxSteps = 1000;

using namespace Acts;
//using Stepper = EigenStepper<ConstantBField>;
using Stepper = EigenStepper<InterpolatedBFieldMap3D>;
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
  bool output = false;
  std::string device;
  std::string bFieldFileName;
  double pT;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "-h") or (arg == "--help")) {
      show_usage(argv[0]);
      return 0;
    } else if (i + 1 < argc) {
      if ((arg == "-t") or (arg == "--tracks")) {
        nTracks = atoi(argv[++i]);
      } else if ((arg == "-p") or (arg == "--pt")) {
        pT = atof(argv[++i]);
      } else if ((arg == "-o") or (arg == "--output")) {
        output = (atoi(argv[++i]) == 1);
      } else if ((arg == "-d") or (arg == "--device")) {
        device = argv[++i];
      } else if ((arg == "-b") or (arg == "--bf-map")) {
        bFieldFileName = argv[++i];
      } else {
        std::cerr << "Unknown argument." << std::endl;
        return 1;
      }
    }
  }

  std::cout << "----- Propgation test of " << nTracks << " tracks on " << device
            << ". Writing results to obj file? " << output << " ----- "
            << std::endl;
 
  InterpolatedBFieldMap3D bField = Options::readBField(bFieldFileName);
  std::cout << "Reading BField and creating a 3D InterpolatedBFieldMap instance done" << std::endl;

  // Construct a stepper with the bField
  Stepper stepper(bField);
  // Construct a propagator
  PropagatorType propagator(stepper);
  // Construct the propagation options object
  PropagatorOptions propOptions;
  propOptions.maxSteps = 1000;
  propOptions.maxStepSize = 1000;

  // Construct random starting track parameters
  std::default_random_engine generator(42);
  std::normal_distribution<double> gauss(0., 1.);
  std::uniform_real_distribution<double> unif(-1.0 * M_PI, M_PI);
  TrackParameters pars[nTracks];
  for (int i = 0; i < nTracks; i++) {
    Vector3D rPos(0.1 * gauss(generator), 0.1 * gauss(generator),
                  0); // Units: mm
    double phi = unif(generator);
    Vector3D rMom(pT * cos(phi), pT * sin(phi),
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

     GPUERRCHK(cudaMalloc(&d_propagator, sizeof(PropagatorType)));
     GPUERRCHK(cudaMalloc(&d_opt, sizeof(PropagatorOptions)));
     GPUERRCHK(cudaMalloc(&d_pars, nTracks * sizeof(TrackParameters)));
     GPUERRCHK(cudaMalloc(&d_ress, nTracks * sizeof(PropResultType)));

    // Copy from host to device
    GPUERRCHK(cudaMemcpy(d_propagator, &propagator, sizeof(propagator),
               cudaMemcpyHostToDevice));
    GPUERRCHK(cudaMemcpy(d_opt, &propOptions, sizeof(PropagatorOptions),
               cudaMemcpyHostToDevice));
    GPUERRCHK(cudaMemcpy(d_pars, pars, nTracks * sizeof(TrackParameters),
               cudaMemcpyHostToDevice));
    GPUERRCHK(cudaMemcpy(d_ress, ress, nTracks * sizeof(PropResultType),
               cudaMemcpyHostToDevice));

    // Run on device
    int threadsPerBlock = 256;
    int blocksPerGrid = (nTracks + threadsPerBlock - 1) / threadsPerBlock;
    propKernel<<<blocksPerGrid, threadsPerBlock>>>(d_propagator, d_pars, d_opt,
                                                   d_ress, nTracks);
  
    GPUERRCHK( cudaPeekAtLastError() );
    GPUERRCHK( cudaDeviceSynchronize() );

    // Copy result from device to host
    GPUERRCHK( cudaMemcpy(ress, d_ress, nTracks * sizeof(PropResultType),
               cudaMemcpyDeviceToHost));

    // Free the memory on device
    GPUERRCHK(cudaFree(d_propagator));
    GPUERRCHK(cudaFree(d_opt));
    GPUERRCHK(cudaFree(d_pars));
    GPUERRCHK(cudaFree(d_ress));
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

  if (output) {
    // Write result to obj file
    std::cout << "Writing yielded " << nTracks << " tracks to obj files..."
              << std::endl;

    for (int it = 0; it < nTracks; it++) {
      PropResultType res = ress[it];
      std::ofstream obj_track;
      std::string fileName =
          device + "_output/Track-" + std::to_string(it) + ".obj";
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
  }

  std::cout << "------------------------  ending  -----------------------"
            << std::endl;

  return 0;
}