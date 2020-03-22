#include "Definitions.hpp"

namespace Acts {
namespace VectorHelpers {
/// @brief Calculates column-wise cross products of a matrix and a vector and
/// stores the result column-wise in a matrix.
///
/// @param [in] m Matrix that will be used for cross products
/// @param [in] v Vector for cross products
/// @return Constructed matrix
__host__ __device__ inline ActsMatrixD<3, 3> cross(const ActsMatrixD<3, 3> &m,
                                                   const Vector3D &v) {
  ActsMatrixD<3, 3> r;
  r.col(0) = m.col(0).cross(v);
  r.col(1) = m.col(1).cross(v);
  r.col(2) = m.col(2).cross(v);

  return r;
}
} // namespace VectorHelpers
} // namespace Acts
