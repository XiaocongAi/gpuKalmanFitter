#include <functional>

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

namespace Acts {

// Template structure to pass to kernel
template <typename T> struct CudaKernelContainer {
  CudaKernelContainer() = default;

  ACTS_DEVICE_FUNC CudaKernelContainer(T *array, size_t size)
      : _array(array), _size(size) {
    for (size_t i = 0; i < _size; i++) {
      if ((_array + i) == nullptr) {
        printf("Nullptr found.\nTerminating.\n");
        // exit(1);
      }
    }
  }

  ACTS_DEVICE_FUNC T *&array() { return _array; }

  ACTS_DEVICE_FUNC size_t size() const { return _size; }

  ACTS_DEVICE_FUNC T &operator[](size_t i) {
    if (i > _size) {
      printf("Index out of bounds\n");
      return _array[0];
    }
    return _array[i];
  }

  ACTS_DEVICE_FUNC T *end() const { return nullptr; }

  template <typename visitor_t>
  ACTS_DEVICE_FUNC const T *find_if(visitor_t &&visitor) const {
    T *matched = nullptr;
    for (size_t i = 0; i < _size; i++) {
      if (visitor(_array[i])) {
        matched = _array + i;
        break;
      }
    }
    return matched;
  }

private:
  T *_array = nullptr;
  size_t _size = 0;
};

} // namespace Acts
