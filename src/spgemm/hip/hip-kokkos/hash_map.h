#ifndef SPGEMM_KOKKOS_HASH_MAP_H
#define SPGEMM_KOKKOS_HASH_MAP_H

#include <hip/hip_runtime.h>
#include "error_macro.hpp"

// tips: array in hashmap allocate in device
template<typename T, typename I>
struct HashMap {
  using dtype = T;
  using csrIdxType = I;
  static constexpr int M = 8;
  static constexpr size_t Capacity = 0x1 << M;
  static constexpr size_t HashMask = Capacity - 1;
  static constexpr int NullValue = -1;

  csrIdxType* ids;
  dtype* values;
  int* begins;
  int* nexts;

  HashMap(csrIdxType* col_idx_ptr, dtype* values_ptr, size_t rows, size_t nnz): 
          ids(col_idx_ptr), values(values_ptr) {
    KOKKOS_HIP_CHECK( hipMalloc((void**) &begins, sizeof(int) * HashMap::Capacity * rows) )
    KOKKOS_HIP_CHECK( hipMemset(begins, HashMap::NullValue, sizeof(int) * HashMap::Capacity * rows) )
    KOKKOS_HIP_CHECK( hipMalloc((void**) &nexts, sizeof(int) * nnz) )
    KOKKOS_HIP_CHECK( hipMemset(nexts, HashMap::NullValue, sizeof(int) * nnz) )
  }
};

#define HASH(T, I, n)   (n & HashMap<T,I>::HashMask)

#endif //SPGEMM_HASH_MAP_