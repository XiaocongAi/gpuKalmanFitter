#pragma once

// All functions callable from CUDA code must be qualified with __device__
#ifdef __CUDACC__
#define ACTS_DEVICE_FUNC __host__ __device__
// We need cuda_runtime.h to ensure that that EIGEN_USING_STD_MATH macro
// works properly on the device side
#include <cuda_runtime.h>
#else
#define ACTS_DEVICE_FUNC
#endif

#include <Eigen/Core>
#include <Eigen/Dense>

namespace Acts {

// Eigen definitions
template <typename T, unsigned int cols>
using ActsRowVector = Eigen::Matrix<T, 1, cols>;

template <unsigned int cols> using ActsRowVectorD = ActsRowVector<double, cols>;

template <typename T, unsigned int rows, unsigned int cols>
using ActsMatrix = Eigen::Matrix<T, rows, cols>;

template <unsigned int rows, unsigned int cols>
using ActsMatrixD = ActsMatrix<double, rows, cols>;

template <unsigned int rows, unsigned int cols>
using ActsMatrixF = ActsMatrix<float, rows, cols>;

template <typename T, unsigned int rows>
using ActsSymMatrix = Eigen::Matrix<T, rows, rows>;

template <unsigned int rows> using ActsSymMatrixD = ActsSymMatrix<double, rows>;

template <unsigned int rows> using ActsSymMatrixF = ActsSymMatrix<float, rows>;

template <typename T, unsigned int rows>
using ActsVector = Eigen::Matrix<T, rows, 1>;

template <typename T> using ActsXVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T> using ActsMatrix3X = Eigen::Matrix<T, 3, Eigen::Dynamic>;

template <typename T, unsigned int rows>
using ActsMatrix3 = Eigen::Matrix<T, 3, rows>;

template <unsigned int rows> using ActsVectorD = ActsVector<double, rows>;

template <unsigned int rows> using ActsVectorF = ActsVector<float, rows>;

using ParValue_t = double;

using Rotation3D = Eigen::Quaternion<double>;
using Translation3D = Eigen::Translation<double, 3>;
using AngleAxis3D = Eigen::AngleAxisd;
using Transform3D = Eigen::Affine3d;
using Vector3D = Eigen::Matrix<double, 3, 1>;
using Vector2D = Eigen::Matrix<double, 2, 1>;
using RotationMatrix3D = Eigen::Matrix<double, 3, 3>;

/// @enum NavigationDirection
/// The navigation direciton is always with
/// respect to a given momentum or direction
enum NavigationDirection : int { backward = -1, forward = 1 };

enum ParDef : unsigned int {
  eLOC_0 = 0, ///< first coordinate in local surface frame
  eLOC_1 = 1, ///< second coordinate in local surface frame
  eLOC_R = eLOC_0,
  eLOC_PHI = eLOC_1,
  eLOC_RPHI = eLOC_0,
  eLOC_Z = eLOC_1,
  eLOC_X = eLOC_0,
  eLOC_Y = eLOC_1,
  eLOC_D0 = eLOC_0,
  eLOC_Z0 = eLOC_1,
  ePHI = 2,    ///< phi direction of momentum in global frame
  eTHETA = 3,  ///< theta direction of momentum in global frame
  eQOP = 4,    ///< charge/momentum for charged tracks, for neutral tracks it is
               /// 1/momentum
  eT = 5,      /// < The time of the particle
  BoundParsDim /// < The local dimensions
};

/// The dimensions of tracks in free coordinates
constexpr unsigned int FreeParsDim = 8;

using ParID_t = ParDef;
using ParValue_t = double;

///
/// Type namings with bound parameters
///

/// Vector of bound parameters
using BoundVector = ActsVector<ParValue_t, BoundParsDim>;
/// Row vector of bound parameters
using BoundRowVector = ActsRowVector<ParValue_t, BoundParsDim>;
/// Matrix of bound-to-bound parameters
using BoundMatrix = ActsMatrix<ParValue_t, BoundParsDim, BoundParsDim>;
/// Symmetrical matrix of bound-to-bound parameters
using BoundSymMatrix = ActsSymMatrix<ParValue_t, BoundParsDim>;

///
/// Type naming with free parameters
///

/// Vector of free track parameters
using FreeVector = ActsVector<ParValue_t, FreeParsDim>;
/// Matrix of free-to-free parameters
using FreeMatrix = ActsMatrix<ParValue_t, FreeParsDim, FreeParsDim>;
/// Symmetrical matrix of free-to-free parameters
using FreeSymMatrix = ActsSymMatrix<ParValue_t, FreeParsDim>;

///
/// Type namings with bound & free parameters
///

/// Matrix of bound-to-free parameters
using BoundToFreeMatrix = ActsMatrix<ParValue_t, FreeParsDim, BoundParsDim>;
/// Matrix of free-to-bound parameters
using FreeToBoundMatrix = ActsMatrix<ParValue_t, BoundParsDim, FreeParsDim>;

} // namespace Acts
