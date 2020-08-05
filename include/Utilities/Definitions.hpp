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

// for GNU: ignore this specific warning, otherwise just include Eigen/Dense
#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#include <Eigen/Dense>
#pragma GCC diagnostic pop
#else
#include <Eigen/Dense>
#endif

#ifdef TRKDETDESCR_USEFLOATPRECISON
using TDD_real_t = float;
#else
using TDD_real_t = double;
#endif

#define TDD_max_bound_value 10e10

namespace Acts {

/// Tolerance for being on Surface
///
/// @note This is intentionally given w/o an explicit unit to avoid having
///       to include the units header unneccessarily. With the native length
///       unit of mm this corresponds to 0.1um.
static constexpr double s_onSurfaceTolerance = 1e-4;

/// Tolerance for not being within curvilinear projection
/// this allows using the same curvilinear frame to eta = 6,
/// validity tested with IntegrationTests/PropagationTest
static constexpr double s_curvilinearProjTolerance = 0.999995;

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
using ActsMatrixD = ActsMatrix<double, rows, cols>;

template <unsigned int rows, unsigned int cols>
using ActsMatrixF = ActsMatrix<float, rows, cols>;

template <typename T, unsigned int rows>
using ActsSymMatrix = Eigen::Matrix<T, rows, rows>;

template <unsigned int rows>
using ActsSymMatrixD = ActsSymMatrix<double, rows>;

template <unsigned int rows>
using ActsSymMatrixF = ActsSymMatrix<float, rows>;

template <typename T, unsigned int rows>
using ActsVector = Eigen::Matrix<T, rows, 1>;

template <unsigned int rows>
using ActsVectorD = ActsVector<double, rows>;

template <unsigned int rows>
using ActsVectorF = ActsVector<float, rows>;

template <typename T, unsigned int cols>
using ActsRowVector = Eigen::Matrix<T, 1, cols>;

template <typename T, unsigned int cols>
using ActsMatrix3 = Eigen::Matrix<T, 3, cols>;

template <unsigned int cols>
using ActsRowVectorD = ActsRowVector<double, cols>;

template <unsigned int cols>
using ActsRowVectorF = ActsRowVector<float, cols>;

template <typename T>
using ActsMatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

using ActsMatrixXd = ActsMatrixX<double>;
using ActsMatrixXf = ActsMatrixX<float>;

template <typename T>
using ActsVectorX = Eigen::Matrix<T, Eigen::Dynamic, 1>;

using ActsVectorXd = ActsVectorX<double>;
using ActsVectorXf = ActsVectorX<float>;

template <typename T>
using ActsRowVectorX = Eigen::Matrix<T, 1, Eigen::Dynamic>;

using ActsRowVectorXd = ActsRowVectorX<double>;
using ActsRowVectorXf = ActsRowVectorX<float>;

template <typename T>
using ActsMatrix3X = Eigen::Matrix<T, 3, Eigen::Dynamic>;

using Rotation3D = Eigen::Quaternion<double>;
using Translation3D = Eigen::Translation<double, 3>;
using AngleAxis3D = Eigen::AngleAxisd;
using Transform3D = Eigen::Affine3d;
using Vector3D = Eigen::Matrix<double, 3, 1>;
using Vector2D = Eigen::Matrix<double, 2, 1>;
using RotationMatrix3D = Eigen::Matrix<double, 3, 3>;

using Rotation3F = Eigen::Quaternion<float>;
using Translation3F = Eigen::Translation<float, 3>;
using AngleAxis3F = Eigen::AngleAxisf;
using Transform3F = Eigen::Affine3f;
using Vector3F = Eigen::Matrix<float, 3, 1>;
using Vector2F = Eigen::Matrix<float, 2, 1>;
using RotationMatrix3F = Eigen::Matrix<float, 3, 3>;

/// axis defintion element for code readability
/// - please use these for access to the member variables if needed, e.g.
///     double z  = position[Acts::eZ];
///     double px = momentum[Acts::ePX];
///
enum AxisDefs : int {
  // position access
  eX = 0,
  eY = 1,
  eZ = 2,
  // momentum access
  ePX = 0,
  ePY = 1,
  ePZ = 2
};

}  // namespace Acts
