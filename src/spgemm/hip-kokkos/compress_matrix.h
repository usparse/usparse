#ifndef SPGEMM_HIP_KOKKOS_BITSET_H
#define SPGEMM_HIP_KOKKOS_BITSET_H

#include "error_macro.hpp"
#include <cstddef>
#include <hip/hip_runtime.h>

template <typename T> struct CompressMatrix {
  using BitSetEleType = unsigned int;
  using IdType = int;
  static constexpr size_t BitsPerByte = 8;
  static constexpr size_t BitsLen = sizeof(BitSetEleType) * BitsPerByte;
  static constexpr BitSetEleType NullValue = 0xffffffff;
  const size_t rows;
  const size_t cols;
  const size_t nnz;

  T *row_ptr;
  IdType *ids_ptr = nullptr;
  BitSetEleType *col_idx_ptr = nullptr;
  int *use_count = nullptr;
  CompressMatrix(size_t m, size_t n, size_t nnz, T *row_ptr)
      : rows(m), cols(n), nnz(nnz) , row_ptr(row_ptr){
    KOKKOS_HIP_CHECK(hipMalloc((void**)(&ids_ptr), nnz * sizeof(IdType)))
    KOKKOS_HIP_CHECK(hipMemset(ids_ptr, CompressMatrix::NullValue, nnz * sizeof(IdType)))
    KOKKOS_HIP_CHECK(hipMalloc((void**)(&col_idx_ptr), nnz * sizeof(BitSetEleType)))
    KOKKOS_HIP_CHECK(hipMemset(col_idx_ptr, 0, nnz * sizeof(BitSetEleType)))
    KOKKOS_HIP_CHECK(hipMalloc((void**)(&use_count), rows * sizeof(int)))
  }

  ~CompressMatrix() {
    if (use_count != nullptr) {
      KOKKOS_HIP_CHECK(hipFree(use_count))
    }
    if (col_idx_ptr != nullptr) {
      KOKKOS_HIP_CHECK(hipFree(col_idx_ptr))
    }
    if (ids_ptr != nullptr) {
      KOKKOS_HIP_CHECK(hipFree(ids_ptr))
    }
    use_count = nullptr;
    col_idx_ptr = nullptr;
    ids_ptr = nullptr;
  }

  void free() {
    if (use_count != nullptr) {
      KOKKOS_HIP_CHECK(hipFree(use_count))
    }
    if (col_idx_ptr != nullptr) {
      KOKKOS_HIP_CHECK(hipFree(col_idx_ptr))
    }
    if (ids_ptr != nullptr) {
      KOKKOS_HIP_CHECK(hipFree(ids_ptr))
    }
    use_count = nullptr;
    col_idx_ptr = nullptr;
    ids_ptr = nullptr;
  }
};

#endif // SPGEMM_HIP_KOKKOS_BITSET_H
