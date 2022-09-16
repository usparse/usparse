#ifndef SPGEMM_HIP_KOKKOS_BITSET_H
#define SPGEMM_HIP_KOKKOS_BITSET_H

#include "../common/check/hip_check.hpp"
#include "align.h"
#include <cstddef>
#include <hip/hip_runtime.h>

template <typename I> struct CompressMatrix {
  using BitSetEleType = unsigned int;
  using IdType = int;
  static constexpr size_t BitsPerByte = 8;
  static constexpr size_t BitsLen = sizeof(BitSetEleType) * BitsPerByte;
  static constexpr BitSetEleType NullValue = 0xffffffff;
  const size_t rows;
  const size_t cols;
  const size_t nnz;

  I *row_ptr;
  IdType *ids_ptr = nullptr;
  BitSetEleType *col_idx_ptr = nullptr;
  int *use_count = nullptr;
  CompressMatrix(size_t m, size_t n, size_t nnz, I *row_ptr, void *temp_buffer)
      : rows(m), cols(n), nnz(nnz), row_ptr(row_ptr) {
    char *ptr = reinterpret_cast<char *>(temp_buffer);

    ids_ptr = reinterpret_cast<IdType *>(ptr);
    ptr += ALIGN_256_BYTES(nnz * sizeof(IdType));

    col_idx_ptr = reinterpret_cast<BitSetEleType *>(ptr);
    ptr += ALIGN_256_BYTES(nnz * sizeof(BitSetEleType));

    use_count = reinterpret_cast<int *>(ptr);
    ptr += ALIGN_256_BYTES(rows * sizeof(int));

#ifdef DEVICE_MALLOC_INITILIZE
    HIP_CHECK(hipMemset(col_idx_ptr, 0, nnz * sizeof(BitSetEleType)))
#endif
  }

  static size_t buffer_size(size_t m, size_t n, size_t nnz) {
    size_t buffer_size = 0;
    buffer_size += ALIGN_256_BYTES(nnz * sizeof(IdType));
    buffer_size += ALIGN_256_BYTES(nnz * sizeof(BitSetEleType));
    buffer_size += ALIGN_256_BYTES(m * sizeof(int));
    return buffer_size;
  }
};

#endif // SPGEMM_HIP_KOKKOS_BITSET_H
