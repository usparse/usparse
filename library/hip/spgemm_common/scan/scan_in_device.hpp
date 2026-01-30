#ifndef COMMON_SCAN_IN_DEVICE_HPP
#define COMMON_SCAN_IN_DEVICE_HPP

#include "common/usparse_utils.hpp"
#include <hip/hip_runtime.h>

namespace common {
  template <typename T, size_t VEC_SIZE> __device__ __forceinline__ void scan_in_vector_device(T *num) {
    static_assert(VEC_SIZE <= __WF_SIZE__, "vector size must less equal then wrap size");
    // thread id in vector
    const int vector_thread_id = threadIdx.x % VEC_SIZE;
    // vector reduce
    for (int j = 1; j < VEC_SIZE; j <<= 1) {
      T tmp = __shfl_up(*num, j, VEC_SIZE);
      if (vector_thread_id >= j) {
        *num += tmp;
      }
    }
  }
} // namespace common
#endif // COMMON_SCAN_IN_DEVICE_HPP