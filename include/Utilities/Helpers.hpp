#pragma once

#include <cmath>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <stdlib.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "Utilities/Definitions.hpp"

namespace Acts {
namespace VectorHelpers {
/// @brief Calculates column-wise cross products of a matrix and a vector and
/// stores the result column-wise in a matrix.
///
/// @param [in] m Matrix that will be used for cross products
/// @param [in] v Vector for cross products
/// @return Constructed matrix
ACTS_DEVICE_FUNC inline ActsMatrixD<3, 3> cross(const ActsMatrixD<3, 3> &m,
                                                const Vector3D &v) {
  ActsMatrixD<3, 3> r;
  r.col(0) = m.col(0).cross(v);
  r.col(1) = m.col(1).cross(v);
  r.col(2) = m.col(2).cross(v);

  return r;
}

/// Calculate radius in the transverse (xy) plane of a vector
/// @tparam Derived Eigen derived concrete type
/// @param v Any vector like Eigen type, static or dynamic
/// @note Will static assert that the number of rows of @p v is at least 2, or
/// in case of dynamic size, will abort execution if that is not the case.
/// @return The transverse radius value.

template <typename Derived>
ACTS_DEVICE_FUNC double perp(const Eigen::MatrixBase<Derived> &v) noexcept {
  constexpr int rows = Eigen::MatrixBase<Derived>::RowsAtCompileTime;
  if (rows != -1) {
    // static size, do compile time check
    static_assert(rows >= 2,
                  "Perp function not valid for vectors not at least 2D");
  } else {
    // dynamic size
    if (v.rows() < 2) {
      printf("Perp function not valid for vectors not at least 2D\n");
      std::abort();
    }
  }
  return std::sqrt(v[0] * v[0] + v[1] * v[1]);
}

/*
/// Calculate the theta angle (longitudinal w.r.t. z axis) of a vector
/// @tparam Derived Eigen derived concrete type
/// @param v Any vector like Eigen type, static or dynamic
/// @note Will static assert that the number of rows of @p v is at least 3, or
/// in case of dynamic size, will abort execution if that is not the case.
/// @return The theta value
template <typename Derived>
double theta(const Eigen::MatrixBase<Derived> &v) noexcept {
  constexpr int rows = Eigen::MatrixBase<Derived>::RowsAtCompileTime;
  if constexpr (rows != -1) {
    // static size, do compile time check
    static_assert(rows >= 3, "Theta function not valid for non-3D vectors.");
  } else {
    // dynamic size
    if (v.rows() < 3) {
      printf("Theta function not valid for non-3D vectors.\n");
      std::abort();
    }
  }

  return std::atan2(std::sqrt(v[0] * v[0] + v[1] * v[1]), v[2]);
}
*/
} // namespace VectorHelpers

namespace MatrixHelpers {

// Sorting vector
template <typename T> void sort(ActsXVector<T> &vec) {
  thrust::sort(thrust::host, vec.data(), vec.data() + vec.size());
  size_t nUnique =
      thrust::unique(thrust::host, vec.data(), vec.data() + vec.size()) -
      vec.data();
  vec.conservativeResize(nUnique);
}

} // namespace MatrixHelpers

} // namespace Acts
