// This file is part of the Acts project.
//
// Copyright (C) 2016-2018 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

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

/// Common scalar (floating point type used for the default algebra types.
///
/// Defaults to `ActsScalar` but can be customized by the user.
#ifdef ACTS_CUSTOM_SCALARTYPE
using ActsScalar = ACTS_CUSTOM_SCALARTYPE;
#else
using ActsScalar = float;
#endif

#include <Eigen/Dense>

#define TDD_max_bound_value 10e10

namespace Acts {

/// Tolerance for being on Surface
///
/// @note This is intentionally given w/o an explicit unit to avoid having
///       to include the units header unneccessarily. With the native length
///       unit of mm this corresponds to 0.1um.
static constexpr ActsScalar s_onSurfaceTolerance = 1e-4;

/// Tolerance for not being within curvilinear projection
/// this allows using the same curvilinear frame to eta = 6,
/// validity tested with IntegrationTests/PropagationTest
static constexpr ActsScalar s_curvilinearProjTolerance = 0.999995;

/// @enum NavigationDirection
/// The navigation direciton is always with
/// respect to a given momentum or direction
enum NavigationDirection : int { backward = -1, forward = 1 };

///  This is a steering enum to tell which material update stage:
/// - preUpdate  : update on approach of a surface
/// - fullUpdate : update when passing a surface
/// - postUpdate : update when leaving a surface
enum MaterialUpdateStage : int {
  preUpdate = -1,
  fullUpdate = 0,
  postUpdate = 1
};

/// @enum NoiseUpdateMode to tell how to deal with noise term in covariance
/// transport
/// - removeNoise: subtract noise term
/// - addNoise: add noise term
enum NoiseUpdateMode : int { removeNoise = -1, addNoise = 1 };

// Eigen definitions
template <typename T, unsigned int rows, unsigned int cols>
using ActsMatrix = Eigen::Matrix<T, rows, cols>;

template <unsigned int rows, unsigned int cols>
using ActsMatrixD = ActsMatrix<ActsScalar, rows, cols>;

template <unsigned int rows, unsigned int cols>
using ActsMatrixF = ActsMatrix<float, rows, cols>;

template <typename T, unsigned int rows>
using ActsSymMatrix = Eigen::Matrix<T, rows, rows>;

template <unsigned int rows>
using ActsSymMatrixD = ActsSymMatrix<ActsScalar, rows>;

template <unsigned int rows> using ActsSymMatrixF = ActsSymMatrix<float, rows>;

template <typename T, unsigned int rows>
using ActsVector = Eigen::Matrix<T, rows, 1>;

template <unsigned int rows> using ActsVectorD = ActsVector<ActsScalar, rows>;

template <unsigned int rows> using ActsVectorF = ActsVector<float, rows>;

template <typename T, unsigned int cols>
using ActsRowVector = Eigen::Matrix<T, 1, cols>;

template <typename T, unsigned int cols>
using ActsMatrix3 = Eigen::Matrix<T, 3, cols>;

template <unsigned int cols>
using ActsRowVectorD = ActsRowVector<ActsScalar, cols>;

template <unsigned int cols> using ActsRowVectorF = ActsRowVector<float, cols>;

template <typename T>
using ActsMatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

using ActsMatrixXd = ActsMatrixX<ActsScalar>;
using ActsMatrixXf = ActsMatrixX<float>;

template <typename T> using ActsVectorX = Eigen::Matrix<T, Eigen::Dynamic, 1>;

using ActsVectorXd = ActsVectorX<ActsScalar>;
using ActsVectorXf = ActsVectorX<float>;

template <typename T>
using ActsRowVectorX = Eigen::Matrix<T, 1, Eigen::Dynamic>;

using ActsRowVectorXd = ActsRowVectorX<ActsScalar>;
using ActsRowVectorXf = ActsRowVectorX<float>;

template <typename T> using ActsMatrix3X = Eigen::Matrix<T, 3, Eigen::Dynamic>;

// coordinate vectors
using Vector2D = ActsVector<ActsScalar, 2>;
using Vector3D = ActsVector<ActsScalar, 3>;
using Vector4D = ActsVector<ActsScalar, 4>;
// symmetric matrices e.g. for coordinate covariance matrices
using SymMatrix2D = ActsSymMatrix<ActsScalar, 2>;
using SymMatrix3D = ActsSymMatrix<ActsScalar, 3>;
using SymMatrix4D = ActsSymMatrix<ActsScalar, 4>;

// pure translation transformations
using Translation2D = Eigen::Translation<ActsScalar, 2>;
using Translation3D = Eigen::Translation<ActsScalar, 3>;
using Translation4D = Eigen::Translation<ActsScalar, 4>;
// linear (rotation) matrices
using RotationMatrix2D = Eigen::Matrix<ActsScalar, 2, 2>;
using RotationMatrix3D = Eigen::Matrix<ActsScalar, 3, 3>;
using RotationMatrix4D = Eigen::Matrix<ActsScalar, 4, 4>;
// pure rotation transformations. only available in 2d and 3d
using Rotation2D = Eigen::Rotation2D<ActsScalar>;
using Rotation3D = Eigen::Quaternion<ActsScalar>;
using AngleAxis3D = Eigen::AngleAxis<ActsScalar>;
// combined affine transformations. types are chosen for better data alignment:
// - 2d affine compact stored as 2x3 matrix
// - 3d affine stored as 4x4 matrix
// - 4d affine compact stored as 4x5 matrix
using Transform2D = Eigen::Transform<ActsScalar, 2, Eigen::AffineCompact>;
using Transform3D = Eigen::Transform<ActsScalar, 3, Eigen::Affine>;
using Transform4D = Eigen::Transform<ActsScalar, 4, Eigen::AffineCompact>;

// Components of coordinate vectors.
///
/// To be used to access coordinate components by named indices instead of magic
/// numbers. This must be a regular `enum` and not a scoped `enum class` to
/// allow implicit conversion to an integer. The enum value are thus visible
/// directly in `namespace Acts`.
///
/// This index enum is not user-configurable (in contrast e.g. to the track
/// parameter index enums) since it must be compatible with varying
/// dimensionality (2d-4d) and other access methods (`.{x,y,z}()` accessors).
enum CoordinateIndices : unsigned int {
  // generic position-like access
  ePos0 = 0,
  ePos1 = 1,
  ePos2 = 2,
  eTime = 3,
  // generic momentum-like access
  eMom0 = ePos0,
  eMom1 = ePos1,
  eMom2 = ePos2,
  eEnergy = eTime,
  // Cartesian spatial coordinates
  eX = ePos0,
  eY = ePos1,
  eZ = ePos2,
};

static constexpr unsigned int s_surfacesSize = 15;

} // namespace Acts
