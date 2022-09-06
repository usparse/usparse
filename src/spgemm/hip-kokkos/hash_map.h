#ifndef SPGEMM_KOKKOS_HASH_MAP_H
#define SPGEMM_KOKKOS_HASH_MAP_H

#include "error_macro.hpp"
#include <hip/hip_runtime.h>

// tips: array in hashmap allocate in device
template <typename T, typename I> struct HashMap {
  static constexpr int M = 10;
  static constexpr size_t Capacity = 0x1 << M;
  static constexpr size_t HashMask = Capacity - 1;
  static constexpr int NullValue = 0xffffffff;

  I *ids;
  T *values;
  int *begins;
  int *nexts;

  HashMap(I *col_idx_ptr, T *values_ptr, size_t rows, size_t nnz) : ids(col_idx_ptr), values(values_ptr) {
    KOKKOS_HIP_CHECK(hipMalloc((void **)&begins, sizeof(int) * HashMap::Capacity * rows))
    KOKKOS_HIP_CHECK(hipMemset(begins, HashMap::NullValue, sizeof(int) * HashMap::Capacity * rows))
    KOKKOS_HIP_CHECK(hipMalloc((void **)&nexts, sizeof(int) * nnz))
    KOKKOS_HIP_CHECK(hipMemset(nexts, HashMap::NullValue, sizeof(int) * nnz))
  }

  ~HashMap() {
    if (begins != nullptr) {
      KOKKOS_HIP_CHECK(hipFree(begins))
    }
    if (nexts != nullptr) {
      KOKKOS_HIP_CHECK(hipFree(nexts))
    }
  }
};

#define HASH(T, I, n, mask) (n & mask)

#endif // SPGEMM_KOKKOS_HASH_MAP_H
