#ifndef USTB_SPARSE_SPGEMM_SPGEMM_NUMERIC_IMPL_HPP
#define USTB_SPARSE_SPGEMM_SPGEMM_NUMERIC_IMPL_HPP

#include <cstddef>
#include <hip/hip_runtime.h>
#include "../common/check/hip_check.hpp"
#include "align.h"
#include "create_bp_impl.hpp"
#include "csr_matrix.h"
#include "spgemm_numeric_flag.hpp"
#include "spgemm_numeric_bp_kernel.hpp"
#include "spgemm_numeric_hash_kernel.hpp"
#include "spgemm_numeric_esc_kernel.hpp"
#include "split_config.h"
#include "split_group_impl.hpp"
#include "split_single_element_impl.hpp"
#include "spgemm_numeric_single_element_kernel.hpp"

class BreakPointNumericPolicy {
public:
  constexpr static size_t CHUNK_SIZE = 2048;
  constexpr static int BLOCK_THREAD_NUM = 512;

  template <typename T, typename I>
  static size_t buffer_size(const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b) {
    size_t buffer_size = 0;
    size_t alloc_size = 0;

    if (matrix_b.cols > CHUNK_SIZE) {
      alloc_size = bp_buffer_size<T, I, CHUNK_SIZE>(matrix_b);
      buffer_size += ALIGN_256_BYTES(alloc_size);
    }

    return buffer_size;
  }

  template <typename T, typename I>
  static void spgemm_numeric(const T alpha, const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b,
                             CsrMatrix<T, I> &matrix_c, void *temp_buffer) {
    if (matrix_b.cols <= CHUNK_SIZE) {
      const int GLOBAL_BLOCK_NUM = matrix_c.rows;
      constexpr int VEC_SIZE = __WF_SIZE__;
      const int BPS_PER_ROW = 1;
      const int BP_IDX_PER_ROW = 1;
      I *bp = matrix_b.row_ptr;
      if (matrix_b.cols > (CHUNK_SIZE / 2)) {
        (spgemm_numeric_block_per_row_sorted_bp_kernel<T, I, CHUNK_SIZE, BLOCK_THREAD_NUM,
                                                       VEC_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
            alpha, matrix_c.rows, matrix_c.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
            matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, matrix_c.row_ptr, matrix_c.col_idx_ptr,
            matrix_c.values_ptr, bp, BPS_PER_ROW, BP_IDX_PER_ROW);
      } else if (matrix_b.cols > (CHUNK_SIZE / 4)) {
        (spgemm_numeric_block_per_row_sorted_bp_kernel<T, I, (CHUNK_SIZE / 2), BLOCK_THREAD_NUM,
                                                       VEC_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
            alpha, matrix_c.rows, matrix_c.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
            matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, matrix_c.row_ptr, matrix_c.col_idx_ptr,
            matrix_c.values_ptr, bp, BPS_PER_ROW, BP_IDX_PER_ROW);
      } else if (matrix_b.cols > (CHUNK_SIZE / 8)) {
        (spgemm_numeric_block_per_row_sorted_bp_kernel<T, I, (CHUNK_SIZE / 4), BLOCK_THREAD_NUM,
                                                       VEC_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
            alpha, matrix_c.rows, matrix_c.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
            matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, matrix_c.row_ptr, matrix_c.col_idx_ptr,
            matrix_c.values_ptr, bp, BPS_PER_ROW, BP_IDX_PER_ROW);
      } else {
        (spgemm_numeric_block_per_row_sorted_bp_kernel<T, I, (CHUNK_SIZE / 8), (CHUNK_SIZE / 8),
                                                       VEC_SIZE>)<<<GLOBAL_BLOCK_NUM, (CHUNK_SIZE / 8)>>>(
            alpha, matrix_c.rows, matrix_c.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
            matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, matrix_c.row_ptr, matrix_c.col_idx_ptr,
            matrix_c.values_ptr, bp, BPS_PER_ROW, BP_IDX_PER_ROW);
      }
    } else {
      const int GLOBAL_BLOCK_NUM = matrix_c.rows;

      const int BPS_PER_ROW = matrix_b.cols / CHUNK_SIZE + (matrix_b.cols % CHUNK_SIZE == 0 ? 0 : 1);
      const int BP_IDX_PER_ROW = BPS_PER_ROW + 1;

      char *ptr = reinterpret_cast<char *>(temp_buffer);
      size_t alloc_size = 0;
      size_t align_size = 0;

      I *bp = reinterpret_cast<I *>(ptr);
      alloc_size = bp_buffer_size<T, I, CHUNK_SIZE>(matrix_b);
      align_size = ALIGN_256_BYTES(alloc_size);
      ptr += align_size;

      create_bp<T, I, CHUNK_SIZE>(matrix_b, bp, BP_IDX_PER_ROW);

      const double sparseness_c =
          static_cast<double>(matrix_c.nnz) / static_cast<double>(matrix_c.rows * matrix_c.cols);

      if (sparseness_c < 0.2) {
        constexpr size_t VEC_SIZE = 16;
        (spgemm_numeric_block_per_row_sorted_bp_kernel<T, I, CHUNK_SIZE, BLOCK_THREAD_NUM,
                                                       VEC_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
            alpha, matrix_c.rows, matrix_c.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
            matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, matrix_c.row_ptr, matrix_c.col_idx_ptr,
            matrix_c.values_ptr, bp, BPS_PER_ROW, BP_IDX_PER_ROW);
      } else {
        constexpr size_t VEC_SIZE = __WF_SIZE__;
        (spgemm_numeric_block_per_row_sorted_bp_kernel<T, I, CHUNK_SIZE, BLOCK_THREAD_NUM,
                                                       VEC_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
            alpha, matrix_c.rows, matrix_c.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
            matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, matrix_c.row_ptr, matrix_c.col_idx_ptr,
            matrix_c.values_ptr, bp, BPS_PER_ROW, BP_IDX_PER_ROW);
      }
    }
  }
};

class HashMapNumericPolicy {
public:
  constexpr static size_t CHUNK_SIZE = 2048;

  template <typename T, typename I>
  static size_t buffer_size(const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b) {
    size_t buffer_size = 0;
    size_t alloc_size = 0;

    alloc_size = split_group_row_size<I>(matrix_a.rows);
    buffer_size += ALIGN_256_BYTES(alloc_size);

    alloc_size = split_group_buffer_size<I>(matrix_a.rows);
    buffer_size += ALIGN_256_BYTES(alloc_size);

    return buffer_size;
  }

  template <typename T, typename I, typename AlternativePolicy = BreakPointNumericPolicy>
  static void spgemm_numeric(const T alpha, const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b,
                             CsrMatrix<T, I> &matrix_c, void *temp_buffer) {
    char *ptr = reinterpret_cast<char *>(temp_buffer);
    size_t alloc_size = 0;
    size_t align_size = 0;

    I *group_row_num_ptr = reinterpret_cast<I *>(ptr);
    alloc_size = split_group_row_size<I>(matrix_c.rows);
    align_size = ALIGN_256_BYTES(alloc_size);
    ptr += align_size;

    I *split_group_temp_buffer = reinterpret_cast<I *>(ptr);
    alloc_size = split_group_buffer_size<I>(matrix_c.rows);
    align_size = ALIGN_256_BYTES(alloc_size);
    ptr += align_size;

   {
      constexpr size_t ROW_NNZ_BEGIN_VALUE = 0;
      int hgroup_one_num[GROUP_ONE_NUM];
      split_one_element<I>(matrix_c.rows, matrix_c.row_ptr, group_row_num_ptr, hgroup_one_num, split_group_temp_buffer, ROW_NNZ_BEGIN_VALUE);
      constexpr size_t N = 2;
      constexpr size_t VEC_SIZE = 128 / N;
      constexpr size_t BLOCK_THREAD_NUM = 256;
      constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
      const size_t GLOBAL_BLOCK_NUM = hgroup_one_num[0] / BLOCK_VEC_NUM + (hgroup_one_num[0] % BLOCK_VEC_NUM == 0 ? 0 : 1);

      spgemm_numeric_vector_per_row_single_element_kernel<T, I, BLOCK_THREAD_NUM, VEC_SIZE>
          <<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
              alpha, matrix_a.rows, matrix_b.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
              matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, group_row_num_ptr, hgroup_one_num[0],
              matrix_c.row_ptr, matrix_c.col_idx_ptr, matrix_c.values_ptr);
   }

    constexpr size_t ROW_NNZ_BEGIN_VALUE = 1;
    int hgroup_num[GROUP_NUM];
    split_group<I>(matrix_c.rows, matrix_c.row_ptr, group_row_num_ptr, hgroup_num, split_group_temp_buffer, ROW_NNZ_BEGIN_VALUE);

    if (hgroup_num[0] > 0) {
      constexpr size_t N = 2;
      constexpr size_t VEC_SIZE = 16 / N;
      constexpr size_t BLOCK_THREAD_NUM = 256;
      constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
      const size_t GLOBAL_BLOCK_NUM = hgroup_num[0] / BLOCK_VEC_NUM + (hgroup_num[0] % BLOCK_VEC_NUM == 0 ? 0 : 1);

      spgemm_numeric_linear_chain_hashmap_vector_per_row_sorted_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM, VEC_SIZE>
          <<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
              alpha, matrix_a.rows, matrix_b.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
              matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, group_row_num_ptr, hgroup_num[0],
              matrix_c.row_ptr, matrix_c.col_idx_ptr, matrix_c.values_ptr);
    }
    if (hgroup_num[1] > 0) {
      constexpr size_t N = 2;
      constexpr size_t VEC_SIZE = 32 / N;
      constexpr size_t BLOCK_THREAD_NUM = 256;
      constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
      const size_t GLOBAL_BLOCK_NUM = hgroup_num[1] / BLOCK_VEC_NUM + (hgroup_num[1] % BLOCK_VEC_NUM == 0 ? 0 : 1);

      spgemm_numeric_linear_chain_hashmap_vector_per_row_sorted_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM, VEC_SIZE>
          <<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
              alpha, matrix_a.rows, matrix_b.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
              matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, group_row_num_ptr + matrix_c.rows,
              hgroup_num[1], matrix_c.row_ptr, matrix_c.col_idx_ptr, matrix_c.values_ptr);
    }
    if (hgroup_num[2] > 0) {
      constexpr size_t N = 2;
      constexpr size_t VEC_SIZE = 128 / N;
      constexpr size_t BLOCK_THREAD_NUM = 256;
      constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
      const size_t GLOBAL_BLOCK_NUM = hgroup_num[2] / BLOCK_VEC_NUM + (hgroup_num[2] % BLOCK_VEC_NUM == 0 ? 0 : 1);

      spgemm_numeric_linear_chain_hashmap_vector_per_row_sorted_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM, VEC_SIZE>
          <<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
              alpha, matrix_a.rows, matrix_b.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
              matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, group_row_num_ptr + matrix_c.rows * 2,
              hgroup_num[2], matrix_c.row_ptr, matrix_c.col_idx_ptr, matrix_c.values_ptr);
    }
    if (hgroup_num[3] > 0) {
      constexpr size_t N = 2;
      constexpr size_t BLOCK_THREAD_NUM = 256 / N;
      constexpr size_t ITEMS_PER_BUCKET = 8;
      const size_t GLOBAL_BLOCK_NUM = hgroup_num[3];

      spgemm_numeric_block_per_row_flash_sorted_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM, ITEMS_PER_BUCKET>
          <<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
              alpha, matrix_a.rows, matrix_b.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
              matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, group_row_num_ptr + matrix_c.rows * 3,
              hgroup_num[3], matrix_c.row_ptr, matrix_c.col_idx_ptr, matrix_c.values_ptr,
              typename FlashSortDispatchTrait<is_single_wf_scan(N, ITEMS_PER_BUCKET, BLOCK_THREAD_NUM)>::Tag{});
    }
    if (hgroup_num[4] > 0) {
      constexpr size_t N = 2;
      constexpr size_t BLOCK_THREAD_NUM = 512 / N;
      constexpr size_t ITEMS_PER_BUCKET = 8;
      const size_t GLOBAL_BLOCK_NUM = hgroup_num[4];

      spgemm_numeric_block_per_row_flash_sorted_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM, ITEMS_PER_BUCKET>
          <<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
              alpha, matrix_a.rows, matrix_b.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
              matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, group_row_num_ptr + matrix_c.rows * 4,
              hgroup_num[4], matrix_c.row_ptr, matrix_c.col_idx_ptr, matrix_c.values_ptr,
              typename FlashSortDispatchTrait<is_single_wf_scan(N, ITEMS_PER_BUCKET, BLOCK_THREAD_NUM)>::Tag{});
    }
    if (hgroup_num[5] > 0) {
      constexpr size_t N = 2;
      constexpr size_t BLOCK_THREAD_NUM = 1024 / N;
      constexpr size_t ITEMS_PER_BUCKET = 8;
      const size_t GLOBAL_BLOCK_NUM = hgroup_num[5];

      spgemm_numeric_block_per_row_flash_sorted_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM, ITEMS_PER_BUCKET>
          <<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
              alpha, matrix_a.rows, matrix_b.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
              matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, group_row_num_ptr + matrix_c.rows * 5,
              hgroup_num[5], matrix_c.row_ptr, matrix_c.col_idx_ptr, matrix_c.values_ptr,
              typename FlashSortDispatchTrait<is_single_wf_scan(N, ITEMS_PER_BUCKET, BLOCK_THREAD_NUM)>::Tag{});
    }
    if (hgroup_num[6] > 0) {
      constexpr size_t N = 2;
      constexpr size_t BLOCK_THREAD_NUM = 2048 / N;
      constexpr size_t ITEMS_PER_BUCKET = 8;
      const size_t GLOBAL_BLOCK_NUM = hgroup_num[6];

      spgemm_numeric_block_per_row_flash_sorted_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM, ITEMS_PER_BUCKET>
          <<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
              alpha, matrix_a.rows, matrix_b.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
              matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, group_row_num_ptr + matrix_c.rows * 6,
              hgroup_num[6], matrix_c.row_ptr, matrix_c.col_idx_ptr, matrix_c.values_ptr,
              typename FlashSortDispatchTrait<is_single_wf_scan(N, ITEMS_PER_BUCKET, BLOCK_THREAD_NUM)>::Tag{});
    }
    if (hgroup_num[7] > 0) {
      const size_t GLOBAL_BLOCK_NUM = hgroup_num[7];
      constexpr size_t BLOCK_THREAD_NUM = 512;
      constexpr size_t VEC_SIZE = __WF_SIZE__;

      const int BPS_PER_ROW = matrix_b.cols / CHUNK_SIZE + (matrix_b.cols % CHUNK_SIZE == 0 ? 0 : 1);
      const int BP_IDX_PER_ROW = BPS_PER_ROW + 1;

      I *bp = nullptr;
      size_t bp_size = bp_buffer_size<T, I, CHUNK_SIZE>(matrix_b);
      HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&bp), ALIGN_256_BYTES(bp_size)))

      create_bp<T, I, CHUNK_SIZE>(matrix_b, bp, BP_IDX_PER_ROW);

      (spgemm_numeric_block_per_row_sorted_bp_kernel<T, I, CHUNK_SIZE, BLOCK_THREAD_NUM,
                                                     VEC_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          alpha, matrix_c.rows, matrix_c.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
          matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, group_row_num_ptr + matrix_c.rows * 7,
          hgroup_num[7], matrix_c.row_ptr, matrix_c.col_idx_ptr, matrix_c.values_ptr, bp, BPS_PER_ROW, BP_IDX_PER_ROW);

      HIP_CHECK(hipFree(bp));
    }
  }
};

class EscNumericPolicy {
public:
  constexpr static size_t CHUNK_SIZE = 2048;

  template <typename T, typename I>
  static size_t buffer_size(const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b) {
    size_t buffer_size = 0;
    size_t alloc_size = 0;

    alloc_size = split_group_row_size<I>(matrix_a.rows);
    buffer_size += ALIGN_256_BYTES(alloc_size);

    alloc_size = split_group_buffer_size<I>(matrix_a.rows);
    buffer_size += ALIGN_256_BYTES(alloc_size);

    return buffer_size;
  }

  template <typename T, typename I>
  static void spgemm_numeric(const T alpha, const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b,
                             CsrMatrix<T, I> &matrix_c, void *temp_buffer) {
    char *ptr = reinterpret_cast<char *>(temp_buffer);
    size_t alloc_size = 0;
    size_t align_size = 0;

    I *group_row_num_ptr = reinterpret_cast<I *>(ptr);
    alloc_size = split_group_row_size<I>(matrix_a.rows);
    align_size = ALIGN_256_BYTES(alloc_size);
    ptr += align_size;

    I *split_group_temp_buffer = reinterpret_cast<I *>(ptr);
    alloc_size = split_group_buffer_size<I>(matrix_c.rows);
    align_size = ALIGN_256_BYTES(alloc_size);
    ptr += align_size;

    int hgroup_num[GROUP_NUM];
    split_group<I>(matrix_c.rows, matrix_c.row_ptr, group_row_num_ptr, hgroup_num, split_group_temp_buffer, 0);

    if (hgroup_num[0] > 0) {
      constexpr size_t N = 2;
      constexpr size_t VEC_SIZE = 16 / N;
      constexpr size_t BLOCK_THREAD_NUM = 256;
      constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
      const size_t GLOBAL_BLOCK_NUM = hgroup_num[0] / BLOCK_VEC_NUM + (hgroup_num[0] % BLOCK_VEC_NUM == 0 ? 0 : 1);

      spgemm_numeric_esc_vector_per_row_sorted_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM, VEC_SIZE>
          <<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
              alpha, matrix_a.rows, matrix_b.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
              matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, group_row_num_ptr, hgroup_num[0],
              matrix_c.row_ptr, matrix_c.col_idx_ptr, matrix_c.values_ptr);
    }
    if (hgroup_num[1] > 0) {
      constexpr size_t N = 2;
      constexpr size_t VEC_SIZE = 32 / N;
      constexpr size_t BLOCK_THREAD_NUM = 256;
      constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
      const size_t GLOBAL_BLOCK_NUM = hgroup_num[1] / BLOCK_VEC_NUM + (hgroup_num[1] % BLOCK_VEC_NUM == 0 ? 0 : 1);

      spgemm_numeric_esc_vector_per_row_sorted_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM, VEC_SIZE>
          <<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
              alpha, matrix_a.rows, matrix_b.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
              matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, group_row_num_ptr + matrix_c.rows,
              hgroup_num[1], matrix_c.row_ptr, matrix_c.col_idx_ptr, matrix_c.values_ptr);
    }
    if (hgroup_num[2] > 0) {
      constexpr size_t N = 2;
      constexpr size_t VEC_SIZE = 128 / N;
      constexpr size_t BLOCK_THREAD_NUM = 256;
      constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
      const size_t GLOBAL_BLOCK_NUM = hgroup_num[2] / BLOCK_VEC_NUM + (hgroup_num[2] % BLOCK_VEC_NUM == 0 ? 0 : 1);

      spgemm_numeric_esc_vector_per_row_sorted_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM, VEC_SIZE>
          <<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
              alpha, matrix_a.rows, matrix_b.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
              matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, group_row_num_ptr + matrix_c.rows * 2,
              hgroup_num[2], matrix_c.row_ptr, matrix_c.col_idx_ptr, matrix_c.values_ptr);
    }
    if (hgroup_num[3] > 0) {
      constexpr size_t N = 2;
      constexpr size_t BLOCK_THREAD_NUM = 256 / N;
      constexpr size_t ITEMS_PER_BUCKET = 8;
      const size_t GLOBAL_BLOCK_NUM = hgroup_num[3];

      spgemm_numeric_esc_block_per_row_sorted_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM, ITEMS_PER_BUCKET>
          <<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
              alpha, matrix_a.rows, matrix_b.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
              matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, group_row_num_ptr + matrix_c.rows * 3,
              hgroup_num[3], matrix_c.row_ptr, matrix_c.col_idx_ptr, matrix_c.values_ptr,
              typename FlashSortDispatchTrait<is_single_wf_scan(N, ITEMS_PER_BUCKET, BLOCK_THREAD_NUM)>::Tag{});
    }
    if (hgroup_num[4] > 0) {
      constexpr size_t N = 2;
      constexpr size_t BLOCK_THREAD_NUM = 512 / N;
      constexpr size_t ITEMS_PER_BUCKET = 8;
      const size_t GLOBAL_BLOCK_NUM = hgroup_num[4];

      spgemm_numeric_esc_block_per_row_sorted_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM, ITEMS_PER_BUCKET>
          <<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
              alpha, matrix_a.rows, matrix_b.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
              matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, group_row_num_ptr + matrix_c.rows * 4,
              hgroup_num[4], matrix_c.row_ptr, matrix_c.col_idx_ptr, matrix_c.values_ptr,
              typename FlashSortDispatchTrait<is_single_wf_scan(N, ITEMS_PER_BUCKET, BLOCK_THREAD_NUM)>::Tag{});
    }
    if (hgroup_num[5] > 0) {
      constexpr size_t N = 2;
      constexpr size_t BLOCK_THREAD_NUM = 1024 / N;
      constexpr size_t ITEMS_PER_BUCKET = 8;
      const size_t GLOBAL_BLOCK_NUM = hgroup_num[5];

      spgemm_numeric_esc_block_per_row_sorted_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM, ITEMS_PER_BUCKET>
          <<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
              alpha, matrix_a.rows, matrix_b.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
              matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, group_row_num_ptr + matrix_c.rows * 5,
              hgroup_num[5], matrix_c.row_ptr, matrix_c.col_idx_ptr, matrix_c.values_ptr,
              typename FlashSortDispatchTrait<is_single_wf_scan(N, ITEMS_PER_BUCKET, BLOCK_THREAD_NUM)>::Tag{});
    }
    if (hgroup_num[6] > 0) {
      constexpr size_t N = 2;
      constexpr size_t BLOCK_THREAD_NUM = 2048 / N;
      constexpr size_t ITEMS_PER_BUCKET = 8;
      const size_t GLOBAL_BLOCK_NUM = hgroup_num[6];

      spgemm_numeric_esc_block_per_row_sorted_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM, ITEMS_PER_BUCKET>
          <<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
              alpha, matrix_a.rows, matrix_b.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
              matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, group_row_num_ptr + matrix_c.rows * 6,
              hgroup_num[6], matrix_c.row_ptr, matrix_c.col_idx_ptr, matrix_c.values_ptr,
              typename FlashSortDispatchTrait<is_single_wf_scan(N, ITEMS_PER_BUCKET, BLOCK_THREAD_NUM)>::Tag{});
    }
    if (hgroup_num[7] > 0) {
      const size_t GLOBAL_BLOCK_NUM = hgroup_num[7];
      constexpr size_t BLOCK_THREAD_NUM = 512;
      constexpr size_t VEC_SIZE = __WF_SIZE__;

      const int BPS_PER_ROW = matrix_b.cols / CHUNK_SIZE + (matrix_b.cols % CHUNK_SIZE == 0 ? 0 : 1);
      const int BP_IDX_PER_ROW = BPS_PER_ROW + 1;

      I *bp = nullptr;
      size_t bp_size = bp_buffer_size<T, I, CHUNK_SIZE>(matrix_b);
      HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&bp), ALIGN_256_BYTES(bp_size)))

      create_bp<T, I, CHUNK_SIZE>(matrix_b, bp, BP_IDX_PER_ROW);

      (spgemm_numeric_block_per_row_sorted_bp_kernel<T, I, CHUNK_SIZE, BLOCK_THREAD_NUM,
                                                     VEC_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          alpha, matrix_c.rows, matrix_c.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
          matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, group_row_num_ptr + matrix_c.rows * 7,
          hgroup_num[7], matrix_c.row_ptr, matrix_c.col_idx_ptr, matrix_c.values_ptr, bp, BPS_PER_ROW, BP_IDX_PER_ROW);

      HIP_CHECK(hipFree(bp));
    }
  }
};

template <typename T, typename I, typename Policy>
void spgemm_numeric(const T alpha, const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b,
                    CsrMatrix<T, I> &matrix_c, void *temp_buffer) {
  Policy::template spgemm_numeric<T, I>(alpha, matrix_a, matrix_b, matrix_c, temp_buffer);
}
#endif // USTB_SPARSE_SPGEMM_SPGEMM_NUMERIC_IMPL_HPP
