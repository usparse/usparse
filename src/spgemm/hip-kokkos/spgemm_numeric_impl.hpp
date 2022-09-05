#ifndef SPGEMM_KOKKOS_SPGEMM_NUMERIC_IMPL_H
#define SPGEMM_KOKKOS_SPGEMM_NUMERIC_IMPL_H

#include "../common/check/hip_check.hpp"
#include "align.h"
#include "build_flag.h"
#include "csr_matrix.h"
#include "spgemm_numeric_kernel.hpp"
#include "split_config.h"
#include "split_group.h"
#include <cstddef>
#include <hip/hip_runtime.h>

#ifdef DEBUG_MODE
#include <iostream>
#endif

template <typename T, typename I>
void spgemm_numeric(const T alpha, const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b,
                    const I row_nnz_max, CsrMatrix<T, I> &matrix_c, void *temp_buffer, const size_t temp_buffer_size) {

#ifdef DEBUG_MODE
  std::cout << "row_nnz_max: " << row_nnz_max << std::endl;
  std::cout << "cols: " << matrix_c.cols << std::endl;
#endif

  bool is_bp_alloc = false;
  bool is_split_row_num_alloc = false;
  bool is_split_buffer_alloc = false;

  char *ptr = reinterpret_cast<char *>(temp_buffer);
  size_t remain_buffer_size = temp_buffer_size;

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

  if (row_nnz_max <= 2048) {
    I *group_row_num_ptr = nullptr;
    {
      const size_t alloc_size = split_group_row_size<I>(matrix_c.rows);
      const size_t align_size = ALIGN_256_BYTES(alloc_size);
      if (remain_buffer_size >= align_size) {
        group_row_num_ptr = reinterpret_cast<I *>(ptr);
        ptr += align_size;
        remain_buffer_size -= align_size;
      } else {
        HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&group_row_num_ptr), alloc_size))
        is_split_row_num_alloc = true;
      }
    }

    I *split_group_temp_buffer = nullptr;
    {
      const size_t alloc_size = split_group_buffer_size<I>(matrix_c.rows);
      const size_t align_size = ALIGN_256_BYTES(alloc_size);
      if (remain_buffer_size >= align_size) {
        split_group_temp_buffer = reinterpret_cast<I *>(ptr);
        ptr += align_size;
        remain_buffer_size -= align_size;
      } else {
        HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&split_group_temp_buffer), alloc_size))
        is_split_buffer_alloc = true;
      }
    }

    int hgroup_num[GROUP_NUM];
    split_group<I>(matrix_c.rows, matrix_c.row_ptr, group_row_num_ptr, reinterpret_cast<int *>(hgroup_num),
                   split_group_temp_buffer);

    if (hgroup_num[0] > 0) {
      constexpr size_t N = 2;
      constexpr size_t VEC_SIZE = 16 / N;
      constexpr size_t BLOCK_THREAD_NUM = 256;
      constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
      const size_t GLOBAL_BLOCK_NUM = hgroup_num[0] / BLOCK_VEC_NUM + (hgroup_num[0] % BLOCK_VEC_NUM == 0 ? 0 : 1);

      (NUMERIC_VECTOR_PER_ROW_ONE_PASS_METHOD)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          alpha, matrix_a.rows, matrix_a.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
          matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, group_row_num_ptr, hgroup_num[0],
          matrix_c.row_ptr, matrix_c.col_idx_ptr, matrix_c.values_ptr);
    }
    if (hgroup_num[1] > 0) {
      constexpr size_t N = 2;
      constexpr size_t VEC_SIZE = 32 / N;
      constexpr size_t BLOCK_THREAD_NUM = 256;
      constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
      const size_t GLOBAL_BLOCK_NUM = hgroup_num[1] / BLOCK_VEC_NUM + (hgroup_num[1] % BLOCK_VEC_NUM == 0 ? 0 : 1);

      (NUMERIC_VECTOR_PER_ROW_ONE_PASS_METHOD)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          alpha, matrix_a.rows, matrix_a.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
          matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, group_row_num_ptr + matrix_c.rows, hgroup_num[1],
          matrix_c.row_ptr, matrix_c.col_idx_ptr, matrix_c.values_ptr);
    }
    if (hgroup_num[2] > 0) {
      constexpr size_t N = 2;
      constexpr size_t VEC_SIZE = 64 / N;
      constexpr size_t BLOCK_THREAD_NUM = 256;
      constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
      const size_t GLOBAL_BLOCK_NUM = hgroup_num[2] / BLOCK_VEC_NUM + (hgroup_num[2] % BLOCK_VEC_NUM == 0 ? 0 : 1);

      (NUMERIC_VECTOR_PER_ROW_ONE_PASS_METHOD)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          alpha, matrix_a.rows, matrix_a.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
          matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, group_row_num_ptr + matrix_c.rows * 2,
          hgroup_num[2], matrix_c.row_ptr, matrix_c.col_idx_ptr, matrix_c.values_ptr);
    }
    if (hgroup_num[3] > 0) {
      constexpr size_t N = 2;
      constexpr size_t BLOCK_THREAD_NUM = 512 / N;
      const size_t GLOBAL_BLOCK_NUM = hgroup_num[3];

      (NUMERIC_BLOCK_PER_ROW_ONE_PASS_METHOD)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          alpha, matrix_a.rows, matrix_a.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
          matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, group_row_num_ptr + matrix_c.rows * 3,
          hgroup_num[3], matrix_c.row_ptr, matrix_c.col_idx_ptr, matrix_c.values_ptr);
    }
    if (hgroup_num[4] > 0) {
      constexpr size_t N = 2;
      constexpr size_t BLOCK_THREAD_NUM = 1024 / N;
      const size_t GLOBAL_BLOCK_NUM = hgroup_num[4];

      (NUMERIC_BLOCK_PER_ROW_ONE_PASS_METHOD)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          alpha, matrix_a.rows, matrix_a.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
          matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, group_row_num_ptr + matrix_c.rows * 4,
          hgroup_num[4], matrix_c.row_ptr, matrix_c.col_idx_ptr, matrix_c.values_ptr);
    }
    if (hgroup_num[5] > 0) {
      constexpr size_t N = 2;
      constexpr size_t BLOCK_THREAD_NUM = 2048 / N;
      const size_t GLOBAL_BLOCK_NUM = hgroup_num[5];

      (NUMERIC_BLOCK_PER_ROW_ONE_PASS_METHOD)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          alpha, matrix_a.rows, matrix_a.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
          matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, group_row_num_ptr + matrix_c.rows * 5,
          hgroup_num[5], matrix_c.row_ptr, matrix_c.col_idx_ptr, matrix_c.values_ptr);
    }

    if (is_split_buffer_alloc) {
      HIP_CHECK(hipFree(split_group_temp_buffer))
    }

    if (is_split_row_num_alloc) {
      HIP_CHECK(hipFree(group_row_num_ptr))
    }
  } else {
    constexpr int CHUNK_SIZE = 2048;
    constexpr int BLOCK_THREAD_NUM = 512;
    const int GLOBAL_BLOCK_NUM = matrix_c.rows;

    const int BPS_PER_ROW = matrix_b.cols / CHUNK_SIZE + (matrix_b.cols % CHUNK_SIZE == 0 ? 0 : 1);
    const int BP_IDX_PER_ROW = BPS_PER_ROW + 1;

    I *bp = nullptr;
    {
      const size_t alloc_size = sizeof(I) * matrix_b.rows * BP_IDX_PER_ROW;
      const size_t align_size = ALIGN_256_BYTES(alloc_size);
      if (remain_buffer_size >= align_size) {
        bp = reinterpret_cast<I *>(ptr);
        ptr += align_size;
        remain_buffer_size -= align_size;
      } else {
        HIP_CHECK(hipMalloc((void **)&bp, alloc_size))
        is_bp_alloc = true;
      }
      HIP_CHECK(hipMemset(bp, 0, alloc_size))
    }

    if (BP_IDX_PER_ROW <= 8) {
      create_bp_wf_per_row_kernel<I, CHUNK_SIZE, 8>
          <<<512, 256>>>(matrix_b.rows, matrix_b.row_ptr, matrix_b.col_idx_ptr, bp, BP_IDX_PER_ROW);
    } else if (BP_IDX_PER_ROW <= 16) {
      create_bp_wf_per_row_kernel<I, CHUNK_SIZE, 16>
          <<<512, 256>>>(matrix_b.rows, matrix_b.row_ptr, matrix_b.col_idx_ptr, bp, BP_IDX_PER_ROW);
    } else {
      create_bp_wf_per_row_kernel<I, CHUNK_SIZE>
          <<<512, 256>>>(matrix_b.rows, matrix_b.row_ptr, matrix_b.col_idx_ptr, bp, BP_IDX_PER_ROW);
    }

    const double sparseness_c = static_cast<double>(matrix_c.nnz) / static_cast<double>(matrix_c.rows * matrix_c.cols);

    if (sparseness_c < 0.2) {
      constexpr int VEC_SIZE = 16;
      (spgemm_numeric_block_per_row_sorted_bp_kernel<T, I, CHUNK_SIZE, BLOCK_THREAD_NUM,
                                                     VEC_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          alpha, matrix_c.rows, matrix_c.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
          matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, matrix_c.row_ptr, matrix_c.col_idx_ptr,
          matrix_c.values_ptr, bp, BPS_PER_ROW, BP_IDX_PER_ROW);
    } else {
      constexpr int VEC_SIZE = 64;
      (spgemm_numeric_block_per_row_sorted_bp_kernel<T, I, CHUNK_SIZE, BLOCK_THREAD_NUM,
                                                     VEC_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          alpha, matrix_c.rows, matrix_c.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
          matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, matrix_c.row_ptr, matrix_c.col_idx_ptr,
          matrix_c.values_ptr, bp, BPS_PER_ROW, BP_IDX_PER_ROW);
    }

    if (is_bp_alloc) {
      HIP_CHECK(hipFree(bp))
    }
  }
}

#endif // SPGEMM_KOKKOS_SPGEMM_NUMERIC_IMPL_H
