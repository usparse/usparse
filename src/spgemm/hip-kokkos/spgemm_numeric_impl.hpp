#ifndef SPGEMM_KOKKOS_SPGEMM_NUMERIC_IMPL_H
#define SPGEMM_KOKKOS_SPGEMM_NUMERIC_IMPL_H

#include "build_flag.h"
#include "csr_matrix.h"
#include "error_macro.hpp"
#include "spgemm_numeric_kernel.hpp"
#include <cstddef>
#include <hip/hip_runtime.h>

#ifdef DEBUG_MODE
#include <iostream>
#endif

template <typename T, typename I>
void spgemm_numeric(const T alpha, const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b,
                    const I max_row_num, CsrMatrix<T, I> &matrix_c, bool &is_sorted) {

  // calculate part of matrix_c
  // part_split[16, 32, 64, 512, 1024, 2048]
#ifdef DEBUG_MODE
  std::cout << "max_row_num: " << max_row_num << std::endl;
  std::cout << "cols: " << matrix_c.cols << std::endl;
#endif

#ifdef NUMERIC_LINEAR_CHAIN_HASHMAP
#define NUMERIC_VECTOR_PER_ROW_ONE_PASS_METHOD                                                                         \
  spgemm_numeric_linear_chain_hashmap_vector_per_row_sorted_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM, VEC_SIZE>
#define NUMERIC_BLOCK_PER_ROW_ONE_PASS_METHOD                                                                          \
  spgemm_numeric_linear_chain_hashmap_block_per_row_sorted_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM>
#endif

#ifndef NUMERIC_LINEAR_CHAIN_HASHMAP
#define NUMERIC_VECTOR_PER_ROW_ONE_PASS_METHOD                                                                         \
  spgemm_numeric_open_address_hashmap_vector_per_row_sorted_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM, VEC_SIZE>
#define NUMERIC_BLOCK_PER_ROW_ONE_PASS_METHOD                                                                          \
  spgemm_numeric_open_address_hashmap_block_per_row_sorted_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM>
#endif

  if (max_row_num < 2048) {
    constexpr size_t GROUPS_NUM = 7;
    bool *row_group_ptr = nullptr;
    KOKKOS_HIP_CHECK(hipMalloc((void **)(&row_group_ptr), sizeof(bool) * matrix_c.rows * GROUPS_NUM))
    KOKKOS_HIP_CHECK(hipMemset(row_group_ptr, false, sizeof(bool) * matrix_c.rows * GROUPS_NUM))
    (row_split_groups_kernel<I>)<<<512, 256>>>(matrix_c.rows, matrix_c.row_ptr, row_group_ptr);
    if (max_row_num > 0) {
      constexpr size_t N = 2;
      constexpr size_t VEC_SIZE = 16 / N;
      constexpr size_t BLOCK_THREAD_NUM = 256;
      constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
      const size_t GLOBAL_BLOCK_NUM = matrix_c.rows / BLOCK_VEC_NUM + (matrix_c.rows % BLOCK_VEC_NUM == 0 ? 0 : 1);

      (NUMERIC_VECTOR_PER_ROW_ONE_PASS_METHOD)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          alpha, matrix_a.rows, matrix_a.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
          matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, row_group_ptr, matrix_c.row_ptr,
          matrix_c.col_idx_ptr, matrix_c.values_ptr);
    }
    if (max_row_num > 16) {
      constexpr size_t N = 2;
      constexpr size_t VEC_SIZE = 32 / N;
      constexpr size_t BLOCK_THREAD_NUM = 256;
      constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
      const size_t GLOBAL_BLOCK_NUM = matrix_c.rows / BLOCK_VEC_NUM + (matrix_c.rows % BLOCK_VEC_NUM == 0 ? 0 : 1);

      (NUMERIC_VECTOR_PER_ROW_ONE_PASS_METHOD)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          alpha, matrix_a.rows, matrix_a.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
          matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, row_group_ptr + matrix_c.rows, matrix_c.row_ptr,
          matrix_c.col_idx_ptr, matrix_c.values_ptr);
    }
    if (max_row_num > 32) {
      constexpr size_t N = 2;
      constexpr size_t VEC_SIZE = 64 / N;
      constexpr size_t BLOCK_THREAD_NUM = 256;
      constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
      const size_t GLOBAL_BLOCK_NUM = matrix_c.rows / BLOCK_VEC_NUM + (matrix_c.rows % BLOCK_VEC_NUM == 0 ? 0 : 1);

      (NUMERIC_VECTOR_PER_ROW_ONE_PASS_METHOD)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          alpha, matrix_a.rows, matrix_a.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
          matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, row_group_ptr + matrix_c.rows * 2,
          matrix_c.row_ptr, matrix_c.col_idx_ptr, matrix_c.values_ptr);
    }
    if (max_row_num > 64) {
      constexpr size_t N = 2;
      constexpr size_t BLOCK_THREAD_NUM = 512 / N;
      const size_t GLOBAL_BLOCK_NUM = matrix_c.rows;

      (NUMERIC_BLOCK_PER_ROW_ONE_PASS_METHOD)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          alpha, matrix_a.rows, matrix_a.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
          matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, row_group_ptr + matrix_c.rows * 3,
          matrix_c.row_ptr, matrix_c.col_idx_ptr, matrix_c.values_ptr);
    }
    if (max_row_num > 512) {
      constexpr size_t N = 2;
      constexpr size_t BLOCK_THREAD_NUM = 1024 / N;
      const size_t GLOBAL_BLOCK_NUM = matrix_c.rows;

      (NUMERIC_BLOCK_PER_ROW_ONE_PASS_METHOD)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          alpha, matrix_a.rows, matrix_a.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
          matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, row_group_ptr + matrix_c.rows * 4,
          matrix_c.row_ptr, matrix_c.col_idx_ptr, matrix_c.values_ptr);
    }
    if (max_row_num > 1024) {
      constexpr size_t N = 2;
      constexpr size_t BLOCK_THREAD_NUM = 2048 / N;
      const size_t GLOBAL_BLOCK_NUM = matrix_c.rows;

      (NUMERIC_BLOCK_PER_ROW_ONE_PASS_METHOD)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          alpha, matrix_a.rows, matrix_a.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
          matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, row_group_ptr + matrix_c.rows * 5,
          matrix_c.row_ptr, matrix_c.col_idx_ptr, matrix_c.values_ptr);
    }
    is_sorted = true;
  } else {
    constexpr int CHUNK_SIZE = 2048;
    // constexpr int VEC_SIZE;
    constexpr int BLOCK_THREAD_NUM = 512;
    const int GLOBAL_BLOCK_NUM = matrix_c.rows;
    I *workspace_b = nullptr;
    KOKKOS_HIP_CHECK(hipMalloc((void **)&workspace_b, sizeof(I) * matrix_a.nnz))
    if((double)matrix_b.nnz/(matrix_b.rows*matrix_b.cols)>0.05){
        (spgemm_numeric_block_per_row_sorted_multipass_kernel<T, I, CHUNK_SIZE, BLOCK_THREAD_NUM, 64>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
        alpha, matrix_c.rows, matrix_c.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
        matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, matrix_c.row_ptr, matrix_c.col_idx_ptr,
        matrix_c.values_ptr,workspace_b);
    }else{
        (spgemm_numeric_block_per_row_sorted_multipass_kernel<T, I, CHUNK_SIZE, BLOCK_THREAD_NUM, 16>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
        alpha, matrix_c.rows, matrix_c.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
        matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, matrix_c.row_ptr, matrix_c.col_idx_ptr,
        matrix_c.values_ptr,workspace_b);
    }
    is_sorted = true;
  }
}

#endif // SPGEMM_KOKKOS_SPGEMM_NUMERIC_IMPL_H
