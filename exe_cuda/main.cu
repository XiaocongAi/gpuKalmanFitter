#include "Geometry/GeometryContext.hpp"
#include "MagneticField/MagneticFieldContext.hpp"
#include "Plugins/BFieldOptions.hpp"
#include "Plugins/BFieldUtils.hpp"
#include "Propagator/EigenStepper.hpp"
#include "Propagator/Propagator.hpp"
#include "Utilities/ParameterDefinitions.hpp"
#include "EventData/TrackParameters.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define GPUERRCHK(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
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
            << "\t-b,--bf-map \tSpecify the path of *.txt for interpolated "
               "BField map\n"
            << std::endl;
}


using namespace Acts;

// Struct for B field
struct ConstantBField {
  ACTS_DEVICE_FUNC static Vector3D getField(const Vector3D & /*field*/) {
    return Vector3D(0., 0., 2.);
  }
};

// Test actor
struct VoidActor{
struct this_result{
 bool status = false;
};
using result_type = this_result;

template <typename propagator_state_t, typename stepper_t>
    void operator()(propagator_state_t& state, const stepper_t& stepper,
                    result_type& result) const {
            return;
    }
};

// Test aborter
struct VoidAborter{
template <typename propagator_state_t, typename stepper_t, typename result_t>
    bool operator()(propagator_state_t& state, const stepper_t& stepper,
                    result_t& result) const {
            return false;
    }
};

//using Stepper = EigenStepper<ConstantBField>;
using Stepper = EigenStepper<InterpolatedBFieldMap3D>;
using PropagatorType = Propagator<Stepper>;
using PropResultType = PropagatorResult<CurvilinearParameters, typename VoidActor::result_type>;
using PropOptionsType = PropagatorOptions<VoidActor, VoidAborter>;

// Device code
__global__ void propKernel(PropagatorType *propagator, CurvilinearParameters *tpars,
                           PropOptionsType *propOptions,
                           PropResultType *propResult, Vector3D *gridValPtr,
                           int N) {
  // Awkwardly make the grid values pointer to point to memeory on device
  // explicitly
  propagator->refStepper().refField().refMapper().refGrid().refValues() =
      gridValPtr;

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N) {
    propResult[i] = propagator->propagate(tpars[i], *propOptions);
  }
}

int main(int argc, char *argv[]) {
  if (argc < 5) {
    show_usage(argv[0]);
    return 1;
  }
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

    // Create a test context
  GeometryContext gctx = GeometryContext();
  MagneticFieldContext mctx = MagneticFieldContext();

  InterpolatedBFieldMap3D bField = Options::readBField(bFieldFileName);
  std::cout
      << "Reading BField and creating a 3D InterpolatedBFieldMap instance done"
      << std::endl;

  // Construct a stepper with the bField
  Stepper stepper(bField);
  // Construct a propagator
  PropagatorType propagator(stepper);
  // Construct the propagation options object
  PropOptionsType propOptions(gctx, mctx); 
  propOptions.maxSteps = 1000;

  // Construct random starting track parameters
  std::default_random_engine generator(42);
  std::normal_distribution<double> gauss(0., 1.);
  std::uniform_real_distribution<double> unif(-1.0 * M_PI, M_PI);
  std::vector<CurvilinearParameters> startPars;
  startPars.reserve(nTracks);
  for (int i = 0; i < nTracks; i++) {
    BoundSymMatrix cov = BoundSymMatrix::Zero();
    cov <<
      0.01, 0., 0., 0., 0., 0.,
      0., 0.01, 0., 0., 0., 0.,
      0., 0., 0.0001, 0., 0., 0.,
      0., 0., 0., 0.0001, 0., 0.,
      0., 0., 0., 0., 0.0001, 0.,
      0., 0., 0., 0.,     0., 1.;

    double q = 1;
    double time =0;
    Vector3D pos(0, 0.1 * gauss(generator), 0.1 * gauss(generator)); // Units: mm
    //double phi = randPhi(generator);
    //double theta = randTheta(generator);
    Vector3D mom(1, 0, 0); // Units: GeV
    
    startPars.emplace_back(cov, pos, mom, q, time);
  }

  // Propagation result
  std::vector<PropResultType> ress;
  ress.reserve(nTracks);

  auto start = std::chrono::high_resolution_clock::now();

  // Running directly on host or offloading to GPU
  bool useGPU = (device == "gpu" ? true : false);
  if (useGPU) {
    // We have to use a really nasty deep reference when dynamic allocation is
    // used for the grid values which cannot be automatically done on GPU?
    auto &grid = propagator.refStepper().refField().refMapper().refGrid();
    // Get the grid size and values (pointer)
    size_t gridSize = grid.size();
    using GridType = std::remove_reference<decltype(grid)>::type;
    using GridValueType = typename GridType::value_type;
    GridValueType* gridValPtr = grid.refValues();

    // Allocate memory on device
    PropagatorType *d_propagator;
    PropOptionsType *d_opt;
    CurvilinearParameters *d_pars;
    PropResultType *d_ress;
    GridValueType *d_gridValPtr;

    GPUERRCHK(cudaMalloc(&d_propagator, sizeof(PropagatorType)));
    GPUERRCHK(cudaMalloc(&d_opt, sizeof(PropOptionsType)));
    GPUERRCHK(cudaMalloc(&d_pars, nTracks * sizeof(CurvilinearParameters)));
    GPUERRCHK(cudaMalloc(&d_ress, nTracks * sizeof(PropResultType)));
    GPUERRCHK(cudaMalloc(&d_gridValPtr, gridSize * sizeof(GridValueType)));

    // Copy from host to device
    GPUERRCHK(cudaMemcpy(d_propagator, &propagator, sizeof(propagator),
                         cudaMemcpyHostToDevice));
    GPUERRCHK(cudaMemcpy(d_opt, &propOptions, sizeof(PropOptionsType),
                         cudaMemcpyHostToDevice));
    GPUERRCHK(cudaMemcpy(d_pars, startPars.data(), nTracks * sizeof(CurvilinearParameters),
                         cudaMemcpyHostToDevice));
    GPUERRCHK(cudaMemcpy(d_ress, ress.data(), nTracks * sizeof(PropResultType),
                         cudaMemcpyHostToDevice));
    GPUERRCHK(cudaMemcpy(d_gridValPtr, gridValPtr, gridSize * sizeof(GridValueType),
                         cudaMemcpyHostToDevice));

    // Run on device
    int threadsPerBlock = 256;
    int blocksPerGrid = (nTracks + threadsPerBlock - 1) / threadsPerBlock;
    propKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_propagator, d_pars, d_opt, d_ress, d_gridValPtr, nTracks);

    GPUERRCHK(cudaPeekAtLastError());
    GPUERRCHK(cudaDeviceSynchronize());

    // Copy result from device to host
    GPUERRCHK(cudaMemcpy(ress.data(), d_ress, nTracks * sizeof(PropResultType),
                         cudaMemcpyDeviceToHost));

    // Free the memory on device
    GPUERRCHK(cudaFree(d_propagator));
    GPUERRCHK(cudaFree(d_opt));
    GPUERRCHK(cudaFree(d_pars));
    GPUERRCHK(cudaFree(d_ress));
    GPUERRCHK(cudaFree(d_gridValPtr));
  } else {
    // Run on host
    #pragma omp parallel for
    for (int it = 0; it < nTracks; it++) {
      ress[it] = propagator.propagate(startPars[it], propOptions);
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

      obj_track.close();
    }
  }

  std::cout << "------------------------  ending  -----------------------"
            << std::endl;

  return 0;
}
