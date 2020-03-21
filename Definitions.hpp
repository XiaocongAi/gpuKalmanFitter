#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

namespace Acts {

// Eigen definitions
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

constexpr unsigned int BoundParsDim = 6;
constexpr unsigned int FreeParsDim = 8;
///
/// Type namings with bound parameters
///

/// Vector of bound parameters
using BoundVector = ActsVector<ParValue_t, BoundParsDim>;
/// Row vector of bound parameters
// using BoundRowVector = ActsRowVector<ParValue_t, BoundParsDim>;
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

} // namespace Acts
