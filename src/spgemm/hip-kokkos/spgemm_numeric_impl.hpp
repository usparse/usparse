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
                    const I max_row_num, CsrMatrix<T, I> &matrix_c, I *temp_buffer, const size_t temp_buffer_size) {

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
  } else {
    constexpr int CHUNK_SIZE = 2048;
    constexpr int BLOCK_THREAD_NUM = 512;
    const int GLOBAL_BLOCK_NUM = matrix_c.rows;

    const int BPS_PER_ROW = matrix_b.cols / CHUNK_SIZE + (matrix_b.cols % CHUNK_SIZE == 0 ? 0 : 1);
    const int BP_IDX_PER_ROW = BPS_PER_ROW + 1;
    const size_t bp_temp_buffer_size = sizeof(I) * matrix_b.rows * BP_IDX_PER_ROW;
    I *bp = nullptr;
    if (temp_buffer_size >= bp_temp_buffer_size) {
      bp = reinterpret_cast<I *>(temp_buffer);
    } else {
      KOKKOS_HIP_CHECK(hipMalloc((void **)&bp, bp_temp_buffer_size))
    }
    KOKKOS_HIP_CHECK(hipMemset(bp, 0, bp_temp_buffer_size))

    if (BP_IDX_PER_ROW < 8) {
      create_bp_wf_per_row_kernel<I, CHUNK_SIZE, 8, __WF_SIZE__>
          <<<512, 256>>>(matrix_b.rows, matrix_b.row_ptr, matrix_b.col_idx_ptr, bp, BP_IDX_PER_ROW);
    } else if (BP_IDX_PER_ROW < 16) {
      create_bp_wf_per_row_kernel<I, CHUNK_SIZE, 16, __WF_SIZE__>
          <<<512, 256>>>(matrix_b.rows, matrix_b.row_ptr, matrix_b.col_idx_ptr, bp, BP_IDX_PER_ROW);
    } else {
      create_bp_wf_per_row_kernel<I, CHUNK_SIZE, __WF_SIZE__>
          <<<512, 256>>>(matrix_b.rows, matrix_b.row_ptr, matrix_b.col_idx_ptr, bp, BP_IDX_PER_ROW);
    }

    const double sparseness_c = static_cast<double>(matrix_c.nnz) / static_cast<double>(matrix_c.rows * matrix_c.cols);

    if (sparseness_c < 0.2) {
      constexpr int VEC_SIZE = 16;
      (spgemm_numeric_block_per_row_sorted_bp_kernel<T, I, CHUNK_SIZE, BLOCK_THREAD_NUM, VEC_SIZE,
                                                     __WF_SIZE__>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          alpha, matrix_c.rows, matrix_c.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
          matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, matrix_c.row_ptr, matrix_c.col_idx_ptr,
          matrix_c.values_ptr, bp, BPS_PER_ROW, BP_IDX_PER_ROW);
    } else {
      constexpr int VEC_SIZE = 64;
      (spgemm_numeric_block_per_row_sorted_bp_kernel<T, I, CHUNK_SIZE, BLOCK_THREAD_NUM, VEC_SIZE,
                                                     __WF_SIZE__>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          alpha, matrix_c.rows, matrix_c.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
          matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, matrix_c.row_ptr, matrix_c.col_idx_ptr,
          matrix_c.values_ptr, bp, BPS_PER_ROW, BP_IDX_PER_ROW);
    }

    if (temp_buffer_size < bp_temp_buffer_size) {
      KOKKOS_HIP_CHECK(hipFree(bp))
    }
  }
}

#endif // SPGEMM_KOKKOS_SPGEMM_NUMERIC_IMPL_H
