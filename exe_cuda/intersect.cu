#include "Geometry/GeometryContext.hpp"
#include "Utilities/ParameterDefinitions.hpp"
#include "Utilities/Units.hpp"
#include "Test/TestHelper.hpp"
#include "Surfaces/ConvexPolygonBounds.hpp"
#include "Surfaces/PlaneSurface.hpp"

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

using namespace Acts;
using SurfaceBoundsType = ConvexPolygonBounds<3>;
using PlaneSurfaceType = PlaneSurface<SurfaceBoundsType>;

template<typename surface_derived_t>
__global__ void intersectKernel(Vector3D position,
                          Vector3D direction,
                          BoundaryCheck bcheck, 
                          const Surface* surfacePtrs, 
                          SurfaceIntersection* intersections, int nSurfaces){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < nSurfaces) {
    intersections[i] = surfacePtrs[i].intersect<surface_derived_t>(GeometryContext(),position, direction, bcheck);
  }
}       

int main(){

  size_t nSurfaces = 37456;

  // 1) malloc for ConvexBounds
  // ConvexBounds must have default constructor 
  SurfaceBoundsType* covexBounds;
  GPUERRCHK(
     cudaMallocManaged(&covexBounds, sizeof(SurfaceBoundsType) * nSurfaces));

  std::vector<Transform3D> transforms;

  std::string surfaceMeshFile = "triangularMesh_generic.csv";
// Read in file and fill values
  std::ifstream surface_file(surfaceMeshFile.c_str(), std::ios::in);
  std::string line;
  std::getline(surface_file, line);
  std::cout<<"cvs header: "<<line<<std::endl;
  double v1_x = 0., v1_y = 0., v1_z = 0., v2_x = 0., v2_y = 0., v2_z = 0., v3_x = 0., v3_y = 0., v3_z = 0.;
  unsigned int isur =0 ;
  while (std::getline(surface_file, line)) {
    if (line.empty() || line[0] == '%' || line[0] == '#' ||
        line.find_first_not_of(' ') == std::string::npos)
      continue;

    std::istringstream tmp(line);
    tmp >> v1_x >> v1_y >> v1_z >> v2_x >> v2_y >> v2_z >> v3_x >> v3_y >> v3_z;
    Acts::Vector3D v1(v1_x,v1_y,v1_z);
    Acts::Vector3D v2(v2_x,v2_y,v2_z);
    Acts::Vector3D v3(v3_x,v3_y,v3_z);
    // Assume the triangular centeroid is the center
    Acts::Vector3D center = (v1+v2+v3)/3.; 
    Acts::Vector3D normal((v1-v2).cross(v1-v3)); 
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
 
    transforms.push_back(std::move(transform)); 
    
    Acts::Vector2D locV1((v1-center).dot(U), (v1-center).dot(V)); 
    Acts::Vector2D locV2((v2-center).dot(U), (v2-center).dot(V)); 
    Acts::Vector2D locV3((v3-center).dot(U), (v3-center).dot(V)); 
   
    ActsMatrix<double, 3, 2> vertices = ActsMatrix<double, 3, 2>::Zero();
    vertices<<(v1-center).dot(U), (v1-center).dot(V), (v2-center).dot(U), (v2-center).dot(V), (v3-center).dot(U), (v3-center).dot(V);
    covexBounds[isur] = ConvexPolygonBounds<3>(vertices);
    isur++;
  }
  surface_file.close();

  // 2) malloc for surfaces
  PlaneSurfaceType *surfaces;
  GPUERRCHK(
      cudaMallocManaged(&surfaces, sizeof(PlaneSurfaceType) * nSurfaces));
  for (unsigned int i = 0; i < nSurfaces; i++) {
    surfaces[i] =
        PlaneSurfaceType(transforms[i], &covexBounds[i]);
  }
  std::cout << "Creating " << nSurfaces << " ConvexBounds plane surfaces"
            << std::endl;

  // 3) malloc for intersections
   SurfaceIntersection* intersections;
   GPUERRCHK(
      cudaMallocManaged(&intersections, sizeof(SurfaceIntersection) * nSurfaces));

    // 4) Pass position, direction, bcheck by value
    Vector3D position(0,0,0);  
    Vector3D direction(1,0,0);  
    BoundaryCheck bcheck(true);

   // Run on device
    int threadsPerBlock = 256;
    int blocksPerGrid = (nSurfaces + threadsPerBlock - 1) / threadsPerBlock;
    intersectKernel<PlaneSurfaceType><<<blocksPerGrid, threadsPerBlock>>>(
                          position,
                          direction,
                          bcheck, 
                          surfaces, 
                          intersections, 
			  nSurfaces); 

    for(unsigned int i =0; i< nSurfaces; i++){
     if(intersections[i].intersection.status == Intersection::Status::onSurface){
      std::cout<<"On surface "<< i << std::endl; 
     }  
    }

return 0;
}
