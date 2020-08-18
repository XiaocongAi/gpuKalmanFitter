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

  using value_type = T;
  using pointer = T *;
  using const_pointer = T const *;
  using reference = T &;
  using const_reference = T const &;
  using iterator_category = std::random_access_iterator_tag;
  using iterator = pointer;
  using const_iterator = const_pointer;

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

  ACTS_DEVICE_FUNC pointer array() { return _array; }
  ACTS_DEVICE_FUNC pointer data() { return _array; }
  ACTS_DEVICE_FUNC const_pointer array() const { return _array; }
  ACTS_DEVICE_FUNC const_pointer data() const { return _array; }

  ACTS_DEVICE_FUNC size_t size() const { return _size; }

  ACTS_DEVICE_FUNC T &operator[](int i) { return _array[i]; }
  ACTS_DEVICE_FUNC T const &operator[](int i) const { return _array[i]; }

  ACTS_DEVICE_FUNC iterator begin() { return _array; }
  ACTS_DEVICE_FUNC const_iterator begin() const { return _array; }
  ACTS_DEVICE_FUNC iterator end() { return _array + _size; }
  ACTS_DEVICE_FUNC const_iterator end() const { return _array + _size; }

  template <typename visitor_t>
  ACTS_DEVICE_FUNC const_iterator find_if(visitor_t &&visitor) const {
    for (auto i = begin(); i != end(); ++i) {
      if (visitor(*i)) {
        return i;
      }
    }
    return end();
  }

  // thrust::find_if

private:
  T *_array = nullptr;
  size_t _size = 0;
};

} // namespace Acts
