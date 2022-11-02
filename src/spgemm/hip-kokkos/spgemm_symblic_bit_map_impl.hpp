#ifndef SPGEMM_KOKKOS_SPGEMM_SYMBOLIC_IMPL_HPP
#define SPGEMM_KOKKOS_SPGEMM_SYMBOLIC_IMPL_HPP

#include "../common/check/hip_check.hpp"
#include "../common/max/max_impl.hpp"
#include "../common/scan/scan_impl.hpp"
#include "align.h"
#include "compress_matrix.h"
#include "csr_matrix.h"
#include "spgemm_compress_impl.hpp"
#include "spgemm_symbolic_kernel.hpp"
#include <algorithm>
#include <cstddef>
#include <hip/hip_runtime.h>
#include <iostream>

template <typename T, typename I, bool use_compress>
size_t spgemm_symbolic_buffer_size(const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b) {
  size_t buffer_size = 0;
  size_t alloc_size = 0;

  alloc_size = common::scan_buffer_size<I>(matrix_a.rows + 1);
  buffer_size += ALIGN_256_BYTES(alloc_size);

  alloc_size = common::max_buffer_size<I>(matrix_a.rows + 1);
  buffer_size += ALIGN_256_BYTES(alloc_size);

  alloc_size = sizeof(I);
  buffer_size += ALIGN_256_BYTES(alloc_size);

  alloc_size = sizeof(I);
  buffer_size += ALIGN_256_BYTES(alloc_size);
  return buffer_size;
}

template <typename T, typename I>
void spgemm_symbolic(const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b, I *row_ptr_c, size_t *hnnz_c,
                     I *hrow_nnz_max, void *temp_buffer) {
  using bitMapType = unsigned long long;
  constexpr size_t BitsLen = sizeof(bitMapType) * 8;
  char *ptr = reinterpret_cast<char *>(temp_buffer);
  size_t alloc_size = 0;
  size_t align_size = 0;

  // ave nnz per row
  // I ave_nnz_per_row_A = matrix_a.nnz / matrix_a.rows + (matrix_a.nnz % matrix_a.rows == 0 ? 0 : 1);
  I ave_nnz_per_row_B = matrix_b.nnz / matrix_b.rows + (matrix_b.nnz % matrix_b.rows == 0 ? 0 : 1);

  // alloc buffer for scan op
  void *scan_temp_buffer = reinterpret_cast<void *>(ptr);
  alloc_size = common::scan_buffer_size<I>(matrix_a.rows + 1);
  align_size = ALIGN_256_BYTES(alloc_size);
  ptr += align_size;

  // alloc buffer for max op
  void *max_temp_buffer = reinterpret_cast<void *>(ptr);
  alloc_size = common::max_buffer_size<I>(matrix_a.rows + 1);
  align_size = ALIGN_256_BYTES(alloc_size);
  ptr += align_size;

  if (ave_nnz_per_row_B <= 16) {
    constexpr size_t VEC_SIZE = 8;
    constexpr size_t BLOCK_THREAD_NUM = 128;
    constexpr size_t BLOCK_WF_NUM = BLOCK_THREAD_NUM / __WF_SIZE__;
    const size_t GLOBAL_BLOCK_NUM = matrix_a.rows / BLOCK_WF_NUM + (matrix_a.rows % BLOCK_WF_NUM == 0 ? 0 : 1);
    const size_t WF_LDS_NUM = matrix_a.rows / BitsLen + (matrix_a.rows % BitsLen == 0 ? 0 : 1);

    spgemm_symbolic_bit_map_wf_per_row<T, I, bitMapType, BLOCK_THREAD_NUM, VEC_SIZE>
        <<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM, sizeof(bitMapType) * BLOCK_WF_NUM * WF_LDS_NUM>>>(
            matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_b.row_ptr, matrix_b.col_idx_ptr, row_ptr_c);
  } else if (ave_nnz_per_row_B <= 32) {
    constexpr size_t VEC_SIZE = 16;
    constexpr size_t BLOCK_THREAD_NUM = 128;
    constexpr size_t BLOCK_WF_NUM = BLOCK_THREAD_NUM / __WF_SIZE__;
    const size_t GLOBAL_BLOCK_NUM = matrix_a.rows / BLOCK_WF_NUM + (matrix_a.rows % BLOCK_WF_NUM == 0 ? 0 : 1);
    const size_t WF_LDS_NUM = matrix_a.rows / BitsLen + (matrix_a.rows % BitsLen == 0 ? 0 : 1);

    spgemm_symbolic_bit_map_wf_per_row<T, I, bitMapType, BLOCK_THREAD_NUM, VEC_SIZE>
        <<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM, sizeof(bitMapType) * BLOCK_WF_NUM * WF_LDS_NUM>>>(
            matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_b.row_ptr, matrix_b.col_idx_ptr, row_ptr_c);
  } else if (ave_nnz_per_row_B <= 64) {
    constexpr size_t VEC_SIZE = 32;
    constexpr size_t BLOCK_THREAD_NUM = 128;
    constexpr size_t BLOCK_WF_NUM = BLOCK_THREAD_NUM / __WF_SIZE__;
    const size_t GLOBAL_BLOCK_NUM = matrix_a.rows / BLOCK_WF_NUM + (matrix_a.rows % BLOCK_WF_NUM == 0 ? 0 : 1);
    const size_t WF_LDS_NUM = matrix_a.rows / BitsLen + (matrix_a.rows % BitsLen == 0 ? 0 : 1);

    spgemm_symbolic_bit_map_wf_per_row<T, I, bitMapType, BLOCK_THREAD_NUM, VEC_SIZE>
        <<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM, sizeof(bitMapType) * BLOCK_WF_NUM * WF_LDS_NUM>>>(
            matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_b.row_ptr, matrix_b.col_idx_ptr, row_ptr_c);
  } else if (ave_nnz_per_row_B <= 128) {
    constexpr size_t VEC_SIZE = 128;
    constexpr size_t BLOCK_THREAD_NUM = 128;
    constexpr size_t BLOCK_WF_NUM = BLOCK_THREAD_NUM / __WF_SIZE__;
    const size_t GLOBAL_BLOCK_NUM = matrix_a.rows / BLOCK_WF_NUM + (matrix_a.rows % BLOCK_WF_NUM == 0 ? 0 : 1);
    const size_t WF_LDS_NUM = matrix_a.rows / BitsLen + (matrix_a.rows % BitsLen == 0 ? 0 : 1);

    spgemm_symbolic_bit_map_wf_per_row<T, I, bitMapType, BLOCK_THREAD_NUM, VEC_SIZE>
        <<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM, sizeof(bitMapType) * BLOCK_WF_NUM * WF_LDS_NUM>>>(
            matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_b.row_ptr, matrix_b.col_idx_ptr, row_ptr_c);
  } else {
    constexpr size_t VEC_SIZE = 8;
    constexpr size_t BLOCK_THREAD_NUM = 256;
    const size_t GLOBAL_BLOCK_NUM = matrix_a.rows;
    const size_t BLOCK_LDS_NUM = matrix_a.rows / BitsLen + (matrix_a.rows % BitsLen == 0 ? 0 : 1);

    spgemm_symbolic_bit_map_block_per_row<T, I, bitMapType, BLOCK_THREAD_NUM, VEC_SIZE>
        <<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM, sizeof(bitMapType) * BLOCK_LDS_NUM>>>(
            matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_b.row_ptr, matrix_b.col_idx_ptr, row_ptr_c);
  }

  // max num in row_ptr
  {
    I *row_nnz_max_ptr = reinterpret_cast<I *>(ptr);
    alloc_size = sizeof(I);
    align_size = ALIGN_256_BYTES(alloc_size);
    ptr += align_size;
    common::max<I>(matrix_a.rows + 1, row_ptr_c, row_nnz_max_ptr, max_temp_buffer);
  }
  // scan row_ptr
  common::scan<I>(matrix_a.rows + 1, row_ptr_c, scan_temp_buffer);
  // set nnz of matrix c
  alloc_size = sizeof(I);
  align_size = ALIGN_256_BYTES(alloc_size);
  hipDeviceSynchronize();
  *hrow_nnz_max = *(reinterpret_cast<I *>(ptr - align_size));
  *hnnz_c = row_ptr_c[matrix_a.rows];
}

#endif // SPGEMM_KOKKOS_SPGEMM_SYMBOLIC_IMPL_HPP
