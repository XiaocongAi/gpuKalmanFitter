#include "Geometry/GeometryContext.hpp"
#include "Surfaces/ConvexPolygonBounds.hpp"
#include "Surfaces/PlaneSurface.hpp"
#include "Test/Helper.hpp"
#include "Utilities/CudaHelper.hpp"
#include "Utilities/ParameterDefinitions.hpp"
#include "Utilities/Units.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace Acts;

using SurfaceBoundsType = ConvexPolygonBounds<3>;
using PlaneSurfaceType = PlaneSurface<SurfaceBoundsType>;

__global__ void intersectKernel(Vector3D position, Vector3D direction,
                                BoundaryCheck bcheck,
                                const PlaneSurfaceType *surfacePtrs,
                                SurfaceIntersection *intersections,
                                bool *status, int nSurfaces, int offset) {
  int i = blockDim.x * blockIdx.x + threadIdx.x + offset;
  if (i < (nSurfaces + offset)) {
    const SurfaceIntersection intersection = surfacePtrs[i].intersect(
        GeometryContext(), position, direction, bcheck);
    if (intersection.intersection.status == Intersection::Status::reachable and
        intersection.intersection.pathLength >= 0) {
      status[i] = true;
      intersections[i] = intersection;
    }
  }
}

__global__ void copyKernel(SurfaceIntersection *allIntersections,
                           SurfaceIntersection *validIntersections,
                           int *intersectionIndices, int nIntersections,
                           int offset) {
  int i = blockDim.x * blockIdx.x + threadIdx.x + offset;
  if (i < (nIntersections + offset)) {
    int indice = intersectionIndices[i];
    validIntersections[i] = allIntersections[indice];
  }
}

int main() {
  int devId = 0;

  cudaDeviceProp prop;
  GPUERRCHK(cudaGetDeviceProperties(&prop, devId));
  printf("Device : %s\n", prop.name);
  GPUERRCHK(cudaSetDevice(devId));

  // This is number of triangular mesh for the TrackML detector sensitive
  // surfaces
  size_t nSurfaces = 37456;

  // Change the number of streams won't have too much impact
  const int threadsPerBlock = 256, nStreams = 4;
  const int threadsPerStream = (nSurfaces + nStreams - 1) / nStreams;
  const int blocksPerGrid_singleStream =
      (nSurfaces + threadsPerBlock - 1) / threadsPerBlock;
  const int blocksPerGrid_multiStream =
      (threadsPerStream + threadsPerBlock - 1) / threadsPerBlock;

  const int boundsBytes = sizeof(SurfaceBoundsType) * nSurfaces;
  const int surfacesBytes = sizeof(PlaneSurfaceType) * nSurfaces;
  const int statusBytes = sizeof(bool) * nSurfaces;
  const int intersectionsBytes = sizeof(SurfaceIntersection) * nSurfaces;
  const int streamIntersectionBytes =
      sizeof(SurfaceIntersection) * threadsPerStream;
  const int streamStatusBytes = sizeof(bool) * threadsPerStream;
  std::cout << "Bounds bytes = " << boundsBytes << std::endl;
  std::cout << "surfaces bytes = " << surfacesBytes << std::endl;
  std::cout << "intersections bytes = " << intersectionsBytes << std::endl;

  // 1) The transforms (they are used to construct the surfaces)
  std::vector<Transform3D> transforms(nSurfaces);

  // 2) malloc for ConvexBounds (unified memory)
  // ConvexBounds must have default constructor
  SurfaceBoundsType *convexBounds;
  GPUERRCHK(cudaMallocManaged(&convexBounds, boundsBytes)); // use unified
                                                            // memory
  // GPUERRCHK(cudaMallocHost((void **)&convexBounds,
  //                         boundsBytes)); // host pinned

  // 3) malloc for surfaces (unified memory)
  PlaneSurfaceType *surfaces;
  GPUERRCHK(cudaMallocManaged(&surfaces, surfacesBytes)); // use unified memory
  // GPUERRCHK(cudaMallocHost((void **)&surfaces,
  //                         surfacesBytes)); // host pinned

  // 4) malloc for intersections (doesn't have to use unified memory)
  SurfaceIntersection *intersections, *d_intersections;
  GPUERRCHK(cudaMallocHost((void **)&intersections,
                           intersectionsBytes)); // host pinned
  GPUERRCHK(
      cudaMalloc((void **)&d_intersections, intersectionsBytes)); // device

  // 5) malloc for intersection status (valid or not)
  bool *status, *d_status;
  GPUERRCHK(cudaMallocHost((void **)&status,
                           statusBytes));                 // host pinned
  GPUERRCHK(cudaMalloc((void **)&d_status, statusBytes)); // device

  // 6) Pass position, direction, bcheck by value
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
  ActsScalar v1_x = 0., v1_y = 0., v1_z = 0., v2_x = 0., v2_y = 0., v2_z = 0.,
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

    ActsMatrix<ActsScalar, 3, 2> vertices =
        ActsMatrix<ActsScalar, 3, 2>::Zero();
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
  // Prefetch the surfaces and bounds to device (not to biase the first test)
  cudaMemPrefetchAsync(convexBounds, boundsBytes, devId, NULL);
  cudaMemPrefetchAsync(surfaces, surfacesBytes, devId, NULL);
  //@note does it help to prefetch non unified memory as well
  //  cudaMemPrefetchAsync(d_intersections, intersectionsBytes, devId, NULL);
  //  cudaMemPrefetchAsync(d_status, statusBytes, devId, NULL);

  // The baseline case - sequential transfer and execute
  memset(intersections, 0, intersectionsBytes);
  GPUERRCHK(cudaEventRecord(startEvent, 0));
  intersectKernel<<<blocksPerGrid_singleStream, threadsPerBlock>>>(
      position, direction, bcheck, surfaces, d_intersections, d_status,
      nSurfaces, 0);
  GPUERRCHK(cudaMemcpy(intersections, d_intersections, intersectionsBytes,
                       cudaMemcpyDeviceToHost));
  GPUERRCHK(cudaEventRecord(stopEvent, 0));
  GPUERRCHK(cudaEventSynchronize(stopEvent));
  GPUERRCHK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  printf("Time for sequential transfer and execute (ms): %f\n", ms);

  // asynchronous version 1: copy all intersections
  memset(intersections, 0, intersectionsBytes);
  GPUERRCHK(cudaEventRecord(startEvent, 0));
  for (int i = 0; i < nStreams; ++i) {
    int offset = i * threadsPerStream;
    intersectKernel<<<blocksPerGrid_multiStream, threadsPerBlock, 0,
                      stream[i]>>>(position, direction, bcheck, surfaces,
                                   d_intersections, d_status, threadsPerStream,
                                   offset);
    GPUERRCHK(cudaEventRecord(stopEvent, stream[i]));
    GPUERRCHK(cudaEventSynchronize(stopEvent));
    GPUERRCHK(cudaMemcpyAsync(&intersections[offset], &d_intersections[offset],
                              streamIntersectionBytes, cudaMemcpyDeviceToHost,
                              stream[i]));
  }
  GPUERRCHK(cudaEventRecord(stopEvent, 0));
  GPUERRCHK(cudaEventSynchronize(stopEvent));
  GPUERRCHK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  printf("Time for asynchronous V1 transfer and execute (ms): %f\n", ms);

  // asynchronous version 2: copy selected intersections
  GPUERRCHK(cudaEventRecord(startEvent, 0));
  for (int i = 0; i < nStreams; ++i) {
    int offset = i * threadsPerStream;
    intersectKernel<<<blocksPerGrid_multiStream, threadsPerBlock, 0,
                      stream[i]>>>(position, direction, bcheck, surfaces,
                                   d_intersections, d_status, threadsPerStream,
                                   offset);
    GPUERRCHK(cudaEventRecord(stopEvent, stream[i]));
    GPUERRCHK(cudaEventSynchronize(stopEvent));
    GPUERRCHK(cudaMemcpyAsync(&status[offset], &d_status[offset],
                              streamStatusBytes, cudaMemcpyDeviceToHost,
                              stream[i]));
  }
  GPUERRCHK(cudaEventRecord(stopEvent, 0));
  GPUERRCHK(cudaEventSynchronize(stopEvent));
  // 7) allocate for the valid intersection indices
  std::vector<int> indices;
  for (unsigned int i = 0; i < nSurfaces; i++) {
    if (status[i]) {
      indices.push_back(i);
    }
  };
  int *d_indices;
  GPUERRCHK(
      cudaMalloc((void **)&d_indices, indices.size() * sizeof(int))); // device
  GPUERRCHK(cudaMemcpy(d_indices, indices.data(), indices.size() * sizeof(int),
                       cudaMemcpyHostToDevice));
  // 8) allocate for the valid intersections
  SurfaceIntersection *validIntersections, *d_validIntersections;
  const int validIntersectionBytes =
      sizeof(SurfaceIntersection) * indices.size();
  GPUERRCHK(cudaMallocHost((void **)&validIntersections,
                           validIntersectionBytes)); // host pinned
  GPUERRCHK(cudaMalloc((void **)&d_validIntersections,
                       validIntersectionBytes)); // device
  // execute kernel to filter intersections
  copyKernel<<<blocksPerGrid_singleStream, threadsPerBlock>>>(
      d_intersections, d_validIntersections, d_indices, indices.size(), 0);
  GPUERRCHK(cudaEventRecord(stopEvent, 0));
  GPUERRCHK(cudaEventSynchronize(stopEvent));
  // Copy valid intersections from device to host
  GPUERRCHK(cudaMemcpy(validIntersections, d_validIntersections,
                       validIntersectionBytes, cudaMemcpyDeviceToHost));
  GPUERRCHK(cudaEventRecord(stopEvent, 0));
  GPUERRCHK(cudaEventSynchronize(stopEvent));
  GPUERRCHK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  printf("Time for asynchronous V2 transfer and execute (ms): %f\n", ms);
  std::cout << "There are " << indices.size() << " reachable surfaces"
            << std::endl;

  // asynchronous version 3:
  //  //loop over copy, loop over kernel, loop over copy
  //  memset(intersections, 0, intersectionsBytes);
  //  GPUERRCHK(cudaEventRecord(startEvent, 0));
  //  for (int i = 0; i < nStreams; ++i) {
  //    int offset = i * threadsPerStream;
  //    intersectKernel
  //        <<<threadsPerStream / threadsPerBlock, threadsPerBlock, 0,
  //        stream[i]>>>(
  //            position, direction, bcheck, surfaces, d_intersections,
  //            d_status, threadsPerStream, offset);
  //  }
  //  for (int i = 0; i < nStreams; ++i) {
  //    int offset = i * threadsPerStream;
  //    GPUERRCHK(cudaMemcpyAsync(&intersections[offset],
  //    &d_intersections[offset],
  //                              streamIntersectionBytes,
  //                              cudaMemcpyDeviceToHost, stream[i]));
  //  }
  //  GPUERRCHK(cudaEventRecord(stopEvent, 0));
  //  GPUERRCHK(cudaEventSynchronize(stopEvent));
  //  GPUERRCHK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  //  printf("Time for asynchronous V2 transfer and execute (ms): %f\n", ms);

  // GPUERRCHK(cudaPeekAtLastError());
  // GPUERRCHK(cudaDeviceSynchronize());

  //  unsigned int nReachable = 0;
  //  std::vector<SurfaceIntersection> reachableIntersections;
  //  for (unsigned int i = 0; i < nSurfaces; i++) {
  //    if (intersections[i].intersection.status ==
  //            Intersection::Status::reachable and
  //        intersections[i].intersection.pathLength >= 0) {
  //      // std::cout<< "Intersection at "
  //      <<intersections[i].intersection.position
  //      // << std::endl;
  //      reachableIntersections.push_back(intersections[i]);
  //      nReachable++;
  //    }
  //  }
  //  std::cout << "There are " << nReachable << " reachable surfaces" <<
  //  std::endl;

  size_t vCounter = 1;

  // write the reachable intersections
  {
    std::ofstream obj_intersections;
    std::string fileName_ = "intersections.obj";
    obj_intersections.open(fileName_.c_str());

    // Initialize the vertex counter
    // for (const auto &intersection : reachableIntersections) {
    for (unsigned int i = 0; i < indices.size(); i++) {
      const auto &pos = validIntersections[i].intersection.position;
      obj_intersections << "v " << pos.x() << " " << pos.y() << " " << pos.z()
                        << "\n";
    }
    // Write out the line - only if we have at least two points created
    // size_t vBreak = reachableIntersections.size();
    size_t vBreak = indices.size();
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
    // for (const auto &intersection : reachableIntersections) {
    for (unsigned int i = 0; i < indices.size(); i++) {
      const auto &surface = validIntersections[i].object;
      const auto &bounds = surface->bounds<PlaneSurfaceType>();
      const auto &vertices = bounds->vertices();
      for (unsigned int i = 0; i < 3; i++) {
        Vector2D local = vertices.block<1, 2>(i, 0).transpose();
        Vector3D global;
        surface->localToGlobal<PlaneSurfaceType>(GeometryContext(), local,
                                                 Vector3D(1, 1, 1), global);
        obj_surfaces << "v " << global.x() << " " << global.y() << " "
                     << global.z() << "\n";
      }
    }
    // for (; vCounter <= reachableIntersections.size() * 3; vCounter += 3)
    for (; vCounter <= indices.size() * 3; vCounter += 3)
      obj_surfaces << "f " << vCounter << " " << vCounter + 1 << " "
                   << vCounter + 2 << '\n';
    obj_surfaces.close();
  }

  cudaFree(d_intersections);
  cudaFree(d_validIntersections);
  cudaFree(d_status);
  cudaFree(d_indices);
  cudaFreeHost(intersections);
  cudaFreeHost(validIntersections);
  cudaFreeHost(status);
  cudaFree(surfaces);
  cudaFree(convexBounds);

  return 0;
}
