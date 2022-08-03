#ifndef SPGEMM_KOKKOS_SPGEMM_NUMERIC_IMPL_H
#define SPGEMM_KOKKOS_SPGEMM_NUMERIC_IMPL_H

#include "csr_matrix.h"
#include "spgemm_numeric_kernel.hpp"
#include <cstddef>
#include <hip/hip_runtime.h>

template <typename T, typename I>
void spgemm_numeric(const T alpha, const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b,
                    const I max_row_num, CsrMatrix<T, I> &matrix_c, bool &is_sorted) {
  constexpr size_t N = 4;
  constexpr size_t LDS_MAP_CAP = 1 << 5;
  constexpr size_t LDS_HASH_MASK = LDS_MAP_CAP - 1;
  constexpr size_t GLOBAL_BLOCK_NUM = 512;
  constexpr size_t BLOCK_THREAD_NUM = 128;
  constexpr size_t VEC_SIZE = 32;
  constexpr size_t WF_SIZE = 64;

  constexpr size_t LDS_HASH_MAP_SIZE = N * BLOCK_THREAD_NUM;
  if (max_row_num > LDS_HASH_MAP_SIZE) {
    HashMap<T, I> hash_map(matrix_c.col_idx_ptr, matrix_c.values_ptr, matrix_c.rows, matrix_c.nnz);
    (spgemm_numeric_vector_per_row_kernel<T, I, N, LDS_MAP_CAP, LDS_HASH_MASK, BLOCK_THREAD_NUM, VEC_SIZE,
                                          WF_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
        alpha, matrix_a.rows, matrix_a.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
        matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, matrix_c.row_ptr, hash_map.begins, hash_map.nexts,
        hash_map.ids, hash_map.values);
    is_sorted = false;
  } else {
    (spgemm_numeric_vector_per_row_sort_kernel<T, I, N, LDS_MAP_CAP, LDS_HASH_MASK, BLOCK_THREAD_NUM, VEC_SIZE,
                                               WF_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
        alpha, matrix_a.rows, matrix_a.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
        matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, matrix_c.row_ptr, matrix_c.col_idx_ptr,
        matrix_c.values_ptr);
    is_sorted = true;
  }
}

#endif // SPGEMM_KOKKOS_SPGEMM_NUMERIC_IMPL_H