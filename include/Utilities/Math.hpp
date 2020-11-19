#include <chrono>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <thread>

// starting point code for matrix inversion:
// https://gist.github.com/tgjones/06ddde4d9f7794d3883a

// max size of the matrix required to inverse
// the GPU needs a size known at compile time to allocate
// space on the stack
#define MAX 6

namespace Acts {

// forward definition
template <typename T> ACTS_DEVICE_FUNC T determinant(T *m, int size);

template <typename T>
ACTS_DEVICE_FUNC void submatrix(T *m, T *result, int size, int row, int col) {
  int rowCnt = 0, colCnt = 0;

  if (size == 2) {
    int idx = size * size - row * size - col - 1;
    *result = *(m + idx);
  } else {
    for (int i = 0; i < size; i++) {
      if (i != row) {
        colCnt = 0;
        for (int j = 0; j < size; j++) {
          if (j != col) {
            *(result + rowCnt * (size - 1) + colCnt) = *(m + i * size + j);
            colCnt++;
          }
        }
        rowCnt++;
      }
    }
  }
}

template <typename T>
ACTS_DEVICE_FUNC T matMinor(T *m, int size, int row, int col) {

  //       T submat[(size-1)*(size-1)]  does not work on GPU;
  //       it only accepts const size allocations
  //       WORKAROUND: so allocate more than we actually use

  T submat[MAX * MAX];
  submatrix(m, (T *)&submat, size, row, col);
  return determinant((T *)&submat, size - 1);
}

template <typename T> ACTS_DEVICE_FUNC T determinant(T *m, int size) {
  T det = 0.0;

  if (size == 1) {
    det = *m;
  } else if (size == 2) {
    det = (*m) * (*(m + size + 1)) - (*(m + 1)) * (*(m + size));
  } else {
    for (int i = 0; i < size; i++) {
      T minor = matMinor(m, size, 0, i);
      T sign = (i % 2 == 1) ? -1.0 : 1.0;
      det += sign * (*(m + i)) * minor;
    }
  }
  return det;
}

template <typename T>
ACTS_DEVICE_FUNC void invert(const ActsMatrixX<T> *em, ActsMatrixX<T> *result) {
  // make sure the matrix is square
  assert(em->rows() == em->cols());
  const int size = em->rows();

  // copy from eigen matrix (column major) to C array (row major)

  // T m[size*size] does not work on GPU;
  // it only accepts const size allocations
  // WORKAROUND: so allocate more than we actually use
  T m[MAX * MAX];
  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++)
      *(m + i * size + j) = em->coeff(i, j);

  T det = determinant((T *)&m, size);
  T invDet = 1.0 / det;

  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++) {
      T minor = matMinor((T *)&m, size, j, i);
      T sign = ((i + j) % 2 == 1) ? -1.0 : 1.0;
      T cofactorM = minor * sign;

      result->coeffRef(i, j) = invDet * cofactorM;
    }
}

template <typename T>
ACTS_DEVICE_FUNC ActsMatrixX<T> calculateInverse(ActsMatrixX<T> m) {
#ifdef __CUDA_ARCH__
  ActsMatrixX<T> result(m.rows(), m.cols());
  invert(&m, &result);
  return result;
#else
  return m.inverse();
#endif
}

} // namespace Acts
