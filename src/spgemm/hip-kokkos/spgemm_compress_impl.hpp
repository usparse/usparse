#ifndef SPGEMM_KOKKOS_SPGEMM_COMPRESS_IMPL_HPP
#define SPGEMM_KOKKOS_SPGEMM_COMPRESS_IMPL_HPP

#include "spgemm_compress_kernel.hpp"
#include <hip/hip_runtime.h>

template <typename T, typename I> CompressMatrix<I> spgemm_compress(const CsrMatrix<T, I> &matrix) {
  using idType = typename CompressMatrix<I>::IdType;
  using bitType = typename CompressMatrix<I>::BitSetEleType;
  CompressMatrix<I> compress_matrix(matrix.rows, matrix.cols, matrix.nnz, matrix.row_ptr);
  const I nnz_per_row = matrix.nnz / matrix.rows;

  constexpr size_t N = 4;
  const size_t GLOBAL_BLOCK_NUM = 512;
  constexpr size_t BLOCK_THREAD_NUM = 256;
  if (nnz_per_row < 8) {
    (compress_matrix_thread_per_row_kernel<I, CompressMatrix<I>::BitsLen>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
        matrix.rows, matrix.row_ptr, matrix.col_idx_ptr, compress_matrix.ids_ptr, compress_matrix.col_idx_ptr,
        compress_matrix.use_count);
  } else if (nnz_per_row < 16) {
    constexpr size_t VEC_SIZE = 16 / N;
    (compress_matrix_vector_per_row_kernel<I, CompressMatrix<I>::BitsLen, BLOCK_THREAD_NUM,
                                           VEC_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
        matrix.rows, matrix.row_ptr, matrix.col_idx_ptr, compress_matrix.ids_ptr, compress_matrix.col_idx_ptr,
        compress_matrix.use_count);
  } else if (nnz_per_row < 32) {
    constexpr size_t VEC_SIZE = 32 / N;
    (compress_matrix_vector_per_row_kernel<I, CompressMatrix<I>::BitsLen, BLOCK_THREAD_NUM,
                                           VEC_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
        matrix.rows, matrix.row_ptr, matrix.col_idx_ptr, compress_matrix.ids_ptr, compress_matrix.col_idx_ptr,
        compress_matrix.use_count);
  } else if (nnz_per_row < 64) {
    constexpr size_t VEC_SIZE = 64 / N;
    (compress_matrix_vector_per_row_kernel<I, CompressMatrix<I>::BitsLen, BLOCK_THREAD_NUM,
                                           VEC_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
        matrix.rows, matrix.row_ptr, matrix.col_idx_ptr, compress_matrix.ids_ptr, compress_matrix.col_idx_ptr,
        compress_matrix.use_count);
  } else if (nnz_per_row < 128) {
    constexpr size_t VEC_SIZE = 128 / N;
    (compress_matrix_vector_per_row_kernel<I, CompressMatrix<I>::BitsLen, BLOCK_THREAD_NUM,
                                           VEC_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
        matrix.rows, matrix.row_ptr, matrix.col_idx_ptr, compress_matrix.ids_ptr, compress_matrix.col_idx_ptr,
        compress_matrix.use_count);
  } else if (nnz_per_row < 256) {
    constexpr size_t VEC_SIZE = 256 / N;
    (compress_matrix_vector_per_row_kernel<I, CompressMatrix<I>::BitsLen, BLOCK_THREAD_NUM,
                                           VEC_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
        matrix.rows, matrix.row_ptr, matrix.col_idx_ptr, compress_matrix.ids_ptr, compress_matrix.col_idx_ptr,
        compress_matrix.use_count);
  } else {
    (compress_matrix_block_per_row_kernel<I, CompressMatrix<I>::BitsLen,
                                          BLOCK_THREAD_NUM>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
        matrix.rows, matrix.row_ptr, matrix.col_idx_ptr, compress_matrix.ids_ptr, compress_matrix.col_idx_ptr,
        compress_matrix.use_count);
  }
  return compress_matrix;
}

#endif // SPGEMM_KOKKOS_SPGEMM_COMPRESS_IMPL_HPP