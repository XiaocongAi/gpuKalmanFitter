#include "Geometry/GeometryContext.hpp"
#include "Surfaces/ConvexPolygonBounds.hpp"
#include "Surfaces/PlaneSurface.hpp"
#include "Test/TestHelper.hpp"
#include "Utilities/ParameterDefinitions.hpp"
#include "Utilities/Units.hpp"
#include "Utilities/CudaHelper.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace Acts;

template< typename object_t>
struct MinimalObjectIntersection{
  /// The object that was (tried to be) intersected
  const object_t* object{nullptr};
  /// Position of the intersection
  Vector3D position{0., 0., 0.};
  /// Signed path length to the intersection (if valid)
  double pathLength{std::numeric_limits<double>::infinity()};
  /// The Status of the intersection
  Intersection::Status status{Intersection::Status::unreachable};
};

using SurfaceBoundsType = ConvexPolygonBounds<3>;
using PlaneSurfaceType = PlaneSurface<SurfaceBoundsType>;
using MinimalSurfaceIntersection = MinimalObjectIntersection<PlaneSurfaceType>;

template <typename surface_derived_t>
__global__ void
intersectKernel(Vector3D position, Vector3D direction, BoundaryCheck bcheck,
                const PlaneSurfaceType *surfacePtrs,
                SurfaceIntersection *intersections, int nSurfaces, int offset) {
  int i = blockDim.x * blockIdx.x + threadIdx.x + offset;
  if (i < (nSurfaces+offset)) {
    intersections[i] = surfacePtrs[i].intersect<surface_derived_t>(
        GeometryContext(), position, direction, bcheck);
  }
}

int main() {
  int devId = 0;

  cudaDeviceProp prop;
  GPUERRCHK(cudaGetDeviceProperties(&prop, devId));
  printf("Device : %s\n", prop.name);
  GPUERRCHK(cudaSetDevice(devId));

  // This is number of triangular mesh for the TrackML detector sensitive surfaces
  size_t nSurfaces = 37456;

  // Change the number of streams won't have too much impact
  const int threadsPerBlock = 256, nStreams = 4;
  const int threadsPerStream = (nSurfaces + nStreams - 1) / nStreams;
  const int blocksPerGrid_singleStream = (nSurfaces + threadsPerBlock - 1) / threadsPerBlock;
  const int blocksPerGrid_multiStream = (threadsPerStream + threadsPerBlock - 1) / threadsPerBlock;

  const int boundsBytes = sizeof(SurfaceBoundsType) * nSurfaces;
  const int surfacesBytes = sizeof(PlaneSurfaceType) * nSurfaces;
  const int intersectionsBytes = sizeof(SurfaceIntersection) * nSurfaces;
  const int streamBytes = sizeof(SurfaceIntersection) * threadsPerStream;
  std::cout<<"Bounds bytes = "<< boundsBytes<<std::endl;
  std::cout<<"surfaces bytes = "<< surfacesBytes<<std::endl;
  std::cout<<"intersections bytes = "<< intersectionsBytes<<std::endl;
  std::cout<<"MinimalIntersections bytes = "<< sizeof(MinimalSurfaceIntersection)*nSurfaces<<std::endl;


  // 1) The transforms (they are used to construct the surfaces)
  std::vector<Transform3D> transforms(nSurfaces);

  // 2) malloc for ConvexBounds (unified memory)
  // ConvexBounds must have default constructor
  SurfaceBoundsType *convexBounds;
  GPUERRCHK(cudaMallocManaged(&convexBounds, boundsBytes)); //use unified memory
  //GPUERRCHK(cudaMallocHost((void **)&convexBounds,
  //                         boundsBytes)); // host pinned

  // 3) malloc for surfaces (unified memory)
  PlaneSurfaceType *surfaces;
  GPUERRCHK(cudaMallocManaged(&surfaces, surfacesBytes)); //use unified memory
  //GPUERRCHK(cudaMallocHost((void **)&surfaces,
  //                         surfacesBytes)); // host pinned

   // 4) malloc for intersections (doesn't have to use unified memory)
  SurfaceIntersection *intersections, *d_intersections;
  GPUERRCHK(cudaMallocHost((void **)&intersections,
                           intersectionsBytes)); // host pinned
  GPUERRCHK(
      cudaMalloc((void **)&d_intersections, intersectionsBytes)); // device

  // 5) Pass position, direction, bcheck by value
  Vector3D position(0, 0, 0);
  Vector3D direction(1, 0.1, 0);
  BoundaryCheck bcheck(true);

  // Fill the transforms and convexBounds on host
  std::string surfaceMeshFile = "triangularMesh_generic.csv";
  // Read in file and fill values
  std::ifstream surface_file(surfaceMeshFile.c_str(), std::ios::in);
  std::string line;
  std::getline(surface_file, line);
  std::cout << "Input header: " << line << std::endl;
  double v1_x = 0., v1_y = 0., v1_z = 0., v2_x = 0., v2_y = 0., v2_z = 0.,
         v3_x = 0., v3_y = 0., v3_z = 0.;
  unsigned int isur = 0;
  while (std::getline(surface_file, line)) {
    if (line.empty() || line[0] == '%' || line[0] == '#' ||
        line.find_first_not_of(' ') == std::string::npos)
      continue;

    std::istringstream tmp(line);
    tmp >> v1_x >> v1_y >> v1_z >> v2_x >> v2_y >> v2_z >> v3_x >> v3_y >> v3_z;
    Acts::Vector3D v1(v1_x, v1_y, v1_z);
    Acts::Vector3D v2(v2_x, v2_y, v2_z);
    Acts::Vector3D v3(v3_x, v3_y, v3_z);
    // Assume the triangular centeroid is the center
    Acts::Vector3D center = (v1 + v2 + v3) / 3.;
    Acts::Vector3D normal = (v1 - v2).cross(v1 - v3);
    Vector3D T = normal.normalized();
    // Assume curvilinear frame as the local frame
    Vector3D U = std::abs(T.dot(Vector3D::UnitZ())) < s_curvilinearProjTolerance
                     ? Vector3D::UnitZ().cross(T).normalized()
                     : Vector3D::UnitX().cross(T).normalized();
    Vector3D V = T.cross(U);

    RotationMatrix3D curvilinearRotation;
    curvilinearRotation.col(0) = U;
    curvilinearRotation.col(1) = V;
    curvilinearRotation.col(2) = T;
    Transform3D transform{curvilinearRotation};
    transform.pretranslate(center);

    transforms[isur] = std::move(transform);

    Acts::Vector2D locV1((v1 - center).dot(U), (v1 - center).dot(V));
    Acts::Vector2D locV2((v2 - center).dot(U), (v2 - center).dot(V));
    Acts::Vector2D locV3((v3 - center).dot(U), (v3 - center).dot(V));

    ActsMatrix<double, 3, 2> vertices = ActsMatrix<double, 3, 2>::Zero();
    vertices << (v1 - center).dot(U), (v1 - center).dot(V),
        (v2 - center).dot(U), (v2 - center).dot(V), (v3 - center).dot(U),
        (v3 - center).dot(V);
    convexBounds[isur] = ConvexPolygonBounds<3>(vertices);
    isur++;
  }
  surface_file.close();

  // Create the surfaces
  for (unsigned int i = 0; i < nSurfaces; i++) {
    surfaces[i] = PlaneSurfaceType(transforms[i], &convexBounds[i]);
  }
  std::cout << "Creating " << nSurfaces << " ConvexBounds plane surfaces"
            << std::endl;

  float ms; // elapsed time in milliseconds

  // Create events and streams
  cudaEvent_t startEvent, stopEvent, dummyEvent;
  cudaStream_t stream[nStreams];
  GPUERRCHK(cudaEventCreate(&startEvent));
  GPUERRCHK(cudaEventCreate(&stopEvent));
  GPUERRCHK(cudaEventCreate(&dummyEvent));
  for (int i = 0; i < nStreams; ++i) {
    GPUERRCHK(cudaStreamCreate(&stream[i]));
  }

  // Run on device
  // The baseline case - sequential transfer and execute
  memset(intersections, 0, intersectionsBytes);
  GPUERRCHK(cudaEventRecord(startEvent, 0));
  intersectKernel<PlaneSurfaceType><<<blocksPerGrid_singleStream, threadsPerBlock>>>(
      position, direction, bcheck, surfaces, d_intersections, nSurfaces, 0);
  GPUERRCHK(cudaMemcpy(intersections, d_intersections, intersectionsBytes,
                       cudaMemcpyDeviceToHost));
  GPUERRCHK(cudaEventRecord(stopEvent, 0));
  GPUERRCHK(cudaEventSynchronize(stopEvent));
  GPUERRCHK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  printf("Time for sequential transfer and execute (ms): %f\n", ms);

  // asynchronous version 1: loop over {kernel, copy}
  memset(intersections, 0, intersectionsBytes);
  GPUERRCHK(cudaEventRecord(startEvent, 0));
  for (int i = 0; i < nStreams; ++i) {
    int offset = i * threadsPerStream;
    intersectKernel<PlaneSurfaceType>
        <<<blocksPerGrid_multiStream, threadsPerBlock, 0, stream[i]>>>(
            position, direction, bcheck, surfaces, d_intersections, threadsPerStream,
            offset);
    GPUERRCHK(cudaMemcpyAsync(&intersections[offset], &d_intersections[offset],
                              streamBytes, cudaMemcpyDeviceToHost, stream[i]));
  }
  GPUERRCHK(cudaEventRecord(stopEvent, 0));
  GPUERRCHK(cudaEventSynchronize(stopEvent));
  GPUERRCHK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  printf("Time for asynchronous V1 transfer and execute (ms): %f\n", ms);

  // asynchronous version 2:
  //loop over copy, loop over kernel, loop over copy
  memset(intersections, 0, intersectionsBytes);
  GPUERRCHK(cudaEventRecord(startEvent, 0));
  for (int i = 0; i < nStreams; ++i) {
    int offset = i * threadsPerStream;
    intersectKernel<PlaneSurfaceType>
        <<<threadsPerStream / threadsPerBlock, threadsPerBlock, 0, stream[i]>>>(
            position, direction, bcheck, surfaces, d_intersections, threadsPerStream,
            offset);
  }
  for (int i = 0; i < nStreams; ++i) {
    int offset = i * threadsPerStream;
    GPUERRCHK(cudaMemcpyAsync(&intersections[offset], &d_intersections[offset],
                              streamBytes, cudaMemcpyDeviceToHost, stream[i]));
  }
  GPUERRCHK(cudaEventRecord(stopEvent, 0));
  GPUERRCHK(cudaEventSynchronize(stopEvent));
  GPUERRCHK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  printf("Time for asynchronous V2 transfer and execute (ms): %f\n", ms);

  // GPUERRCHK(cudaPeekAtLastError());
  // GPUERRCHK(cudaDeviceSynchronize());

  unsigned int nReachable = 0;
  std::vector<SurfaceIntersection> reachableIntersections;
  for (unsigned int i = 0; i < nSurfaces; i++) {
    if (intersections[i].intersection.status ==
            Intersection::Status::reachable and
        intersections[i].intersection.pathLength >= 0) {
      // std::cout<< "Intersection at " <<intersections[i].intersection.position
      // << std::endl;
      reachableIntersections.push_back(intersections[i]);
      nReachable++;
    }
  }
  std::cout << "There are " << nReachable << " reachable surfaces" << std::endl;

  size_t vCounter = 1;

  // write the reachable intersections
  {
    std::ofstream obj_intersections;
    std::string fileName_ = "intersections.obj";
    obj_intersections.open(fileName_.c_str());

    // Initialize the vertex counter
    for (const auto &intersection : reachableIntersections) {
      const auto &pos = intersection.intersection.position;
      obj_intersections << "v " << pos.x() << " " << pos.y() << " " << pos.z()
                        << "\n";
    }
    // Write out the line - only if we have at least two points created
    size_t vBreak = reachableIntersections.size();
    for (; vCounter < vBreak; ++vCounter) {
      obj_intersections << "l " << vCounter << " " << vCounter + 1 << '\n';
    }
    obj_intersections.close();
  }

  // write the reachable surfaces
  {
    std::ofstream obj_surfaces;
    std::string fileName_s = "surfaces.obj";
    obj_surfaces.open(fileName_s.c_str());

    vCounter = 1;
    // Initialize the vertex counter
    for (const auto &intersection : reachableIntersections) {
      const auto &surface = intersection.object;
      const auto &bounds = surface->bounds<PlaneSurfaceType>();
      const auto &vertices = bounds->vertices();
      for (unsigned int i = 0; i < 3; i++) {
        Vector2D local = vertices.block<1, 2>(i, 0).transpose();
        Vector3D global;
        surface->localToGlobal(GeometryContext(), local, Vector3D(1, 1, 1),
                               global);
        obj_surfaces << "v " << global.x() << " " << global.y() << " "
                     << global.z() << "\n";
      }
    }
    for (; vCounter <= reachableIntersections.size() * 3; vCounter += 3)
      obj_surfaces << "f " << vCounter << " " << vCounter + 1 << " "
                   << vCounter + 2 << '\n';
    obj_surfaces.close();
  }

  
  cudaFree(d_intersections);
  cudaFreeHost(intersections);
  cudaFree(surfaces);
  cudaFree(convexBounds);

  return 0;
}
