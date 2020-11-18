#include <chrono>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <thread>

// starting point code for matrix inversion:
// https://gist.github.com/tgjones/06ddde4d9f7794d3883a

namespace Acts {

// forward definition
template <typename T>
ACTS_DEVICE_FUNC T determinant(T** m, int size);

template <typename T>
ACTS_DEVICE_FUNC void submatrix(T** m, T** result, int size, int row, int col) {
       int rowCnt = 0, colCnt = 0;

        // TODO: optimize If-then-else for GPU
        if (size == 2) {
                int idx = abs(size-row-1);
                result[0][0] = m[idx][idx];
        } else {
                for (int i = 0; i < size; i++) {
                        if (i != row) {
                                colCnt = 0;
                                for (int j = 0; j < size; j++) {
                                        if (j != col) {
                                                result[rowCnt][colCnt] = m[i][j];
                                                colCnt++;
                                        }
                                }
                                rowCnt++;
                        }
                }
        }

}

template <typename T>
ACTS_DEVICE_FUNC T matMinor(T** m, int size, int row, int col) {
        T** submat = (T**)malloc(sizeof(T)*(size-1));
        for (int i = 0; i < size-1; i++)
                submat[i] = (T*)malloc(sizeof(T)*(size-1));

        submatrix(m, submat, size, row, col);
        T det = determinant(submat, size-1);

        for (int i = 0; i < size-1; i++)
                free(submat[i]);
        free(submat);
        return det;
}

template <typename T>
ACTS_DEVICE_FUNC T determinant(T** m, int size) {
        T det = 0.0;

        // TODO optimize If-Then-Else for GPU
        if (size == 1) {
                det = m[0][0];
        }
        else if (size == 2) {
                det = m[0][0]*m[1][1] - m[0][1]*m[1][0];
        }
        else {
                for (int i = 0; i < size; i++) {
                        double minor = matMinor(m, size, 0, i);
                        double sign = (i % 2 == 1) ? -1.0 : 1.0;
                        det += sign * m[0][i] * minor;
                }
        }
        return det;
}

template <typename T>
ACTS_DEVICE_FUNC void invert(const ActsMatrixX<T>* em, ActsMatrixX<T> *result) {
        // make sure the matrix is square
        assert(em->rows() == em->cols());
        int size = em->rows();

        // copy from eigen matrix (column major) to C array (row major)
        T** m = (T**)malloc(sizeof(T)*size);
        for (int i = 0; i < size; i++) {
                m[i] = (T*)malloc(sizeof(T)*size);
                for (int j = 0; j < size; j++)
                        m[i][j] = em->coeff(i,j);
        }

        T det = determinant(m, size);
        T invDet = 1.0/det;
//      std::cout << "det=" << det << " ,invDet= " << invDet << std::endl;

        for (int i = 0; i < size; i++)
                for (int j = 0; j < size; j++) {
                        T minor = matMinor(m, size, j, i);
                        T sign = ((i + j) % 2 == 1) ? -1.0 : 1.0;
                        T cofactorM = minor * sign;

                        result->coeffRef(i,j) = invDet * cofactorM;
                }
        for (int i = 0; i < size; i++)
                free(m[i]);
        free(m);
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
