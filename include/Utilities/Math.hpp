#include <chrono>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <thread>

// starting point code for matrix inversion:
// https://gist.github.com/tgjones/06ddde4d9f7794d3883a

namespace Acts {

// forward definition
ACTS_DEVICE_FUNC double determinant(Eigen::MatrixXd m);

ACTS_DEVICE_FUNC Eigen::MatrixXd submatrix(Eigen::MatrixXd m, int row,
                                           int col) {
  int rowCnt = 0, colCnt = 0;
  const int size = m.rows() - 1;

  Eigen::MatrixXd result(size, size);

  // TODO: optimize If-then-else for GPU
  if (size == 1) {
    result(0, 0) = m(abs(m.rows() - row - 1), abs(m.cols() - col - 1));
    return result;
  }

  for (int i = 0; i < m.rows(); i++) {
    if (i != row) {
      colCnt = 0;
      for (int j = 0; j < m.cols(); j++) {
        if (j != col) {
          result(rowCnt, colCnt) = m(i, j);
          colCnt++;
        }
      }
      rowCnt++;
    }
  }
  return result;
}

ACTS_DEVICE_FUNC double matMinor(Eigen::MatrixXd m, int row, int col) {
  Eigen::MatrixXd minorSubmatrix = submatrix(m, row, col);
  return determinant(minorSubmatrix);
}

ACTS_DEVICE_FUNC double determinant(Eigen::MatrixXd m) {
  double det = 0.0;

  // TODO optimize If-Then-Else for GPU
  if (m.rows() == 1 && m.cols() == 1) {
    det = m(0, 0);
  } else if (m.rows() == 2 && m.cols() == 2) {
    det = (m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0));
  } else {
    for (int i = 0; i < m.rows(); i++) {
      double minor = matMinor(m, 0, i);
      double sign = (i % 2 == 1) ? -1.0 : 1.0;
      det += sign * m(0, i) * minor;
    }
  }

  return det;
}

ACTS_DEVICE_FUNC Eigen::MatrixXd invert(const Eigen::MatrixXd m) {

  double det = determinant(m);
  double invDet = 1.0 / det;
  //	std::cout << "det=" << det << " ,invDet= " << invDet << std::endl;

  Eigen::MatrixXd result(m.rows(), m.cols());

  for (int j = 0; j < m.cols(); j++)
    for (int i = 0; i < m.rows(); i++) {
      double minor = matMinor(m, j, i);
      double sign = ((i + j) % 2 == 1) ? -1.0 : 1.0;
      double cofactorM = minor * sign;
      result(i, j) = invDet * cofactorM;
    }

  return result;
}

ACTS_DEVICE_FUNC Eigen::MatrixXd calculateInverse(Eigen::MatrixXd m) {
#ifdef __CUDA_ARCH__
  return invert(m);
#else
  return m.inverse();
#endif
}

} // namespace Acts
