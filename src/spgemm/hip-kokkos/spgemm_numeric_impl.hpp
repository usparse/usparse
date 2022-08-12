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
  constexpr size_t WF_SIZE = 64;

  // calculate part of matrix_c
  // part_split[16, 32, 64, 512, 1024, 2048]
#ifdef DEBUG_MODE
  std::cout << "max_row_num: " << max_row_num << std::endl;
  std::cout << "cols: " << matrix_c.cols << std::endl;
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

      (spgemm_numeric_vector_per_row_sort_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM, VEC_SIZE,
                                                          WF_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
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

      (spgemm_numeric_vector_per_row_sort_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM, VEC_SIZE,
                                                          WF_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
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

      (spgemm_numeric_vector_per_row_sort_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM, VEC_SIZE,
                                                          WF_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          alpha, matrix_a.rows, matrix_a.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
          matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, row_group_ptr + matrix_c.rows * 2,
          matrix_c.row_ptr, matrix_c.col_idx_ptr, matrix_c.values_ptr);
    }
    if (max_row_num > 64) {
      constexpr size_t N = 2;
      constexpr size_t BLOCK_THREAD_NUM = 512 / N;
      const size_t GLOBAL_BLOCK_NUM = matrix_c.rows;

      (spgemm_numeric_block_per_row_sort_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM,
                                                         WF_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          alpha, matrix_a.rows, matrix_a.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
          matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, row_group_ptr + matrix_c.rows * 3,
          matrix_c.row_ptr, matrix_c.col_idx_ptr, matrix_c.values_ptr);
    }
    if (max_row_num > 512) {
      constexpr size_t N = 2;
      constexpr size_t BLOCK_THREAD_NUM = 1024 / N;
      const size_t GLOBAL_BLOCK_NUM = matrix_c.rows;

      (spgemm_numeric_block_per_row_sort_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM,
                                                         WF_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          alpha, matrix_a.rows, matrix_a.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
          matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, row_group_ptr + matrix_c.rows * 4,
          matrix_c.row_ptr, matrix_c.col_idx_ptr, matrix_c.values_ptr);
    }
    if (max_row_num > 1024) {
      constexpr size_t N = 2;
      constexpr size_t BLOCK_THREAD_NUM = 2048 / N;
      const size_t GLOBAL_BLOCK_NUM = matrix_c.rows;

      (spgemm_numeric_block_per_row_sort_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM,
                                                         WF_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          alpha, matrix_a.rows, matrix_a.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
          matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, row_group_ptr + matrix_c.rows * 5,
          matrix_c.row_ptr, matrix_c.col_idx_ptr, matrix_c.values_ptr);
    }
    // if (max_row_num > 2048) {
    //   constexpr size_t N = 4;
    //   constexpr size_t BLOCK_THREAD_NUM = 4096 / N;
    //   const size_t GLOBAL_BLOCK_NUM = matrix_c.rows;

    //   (spgemm_numeric_block_per_row_sort_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM,
    //                                                      WF_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
    //       alpha, matrix_a.rows, matrix_a.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
    //       matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, row_group_ptr + matrix_c.rows * 6,
    //       matrix_c.row_ptr, matrix_c.col_idx_ptr, matrix_c.values_ptr);
    // }
    // KOKKOS_HIP_CHECK(hipFree(row_group_ptr));
    is_sorted = true;
  } else {
    constexpr int CHUNK_SIZE = 2048;
    constexpr int VEC_SIZE = 16;
    constexpr int BLOCK_THREAD_NUM = 512;
    const int GLOBAL_BLOCK_NUM = matrix_c.rows;
    I *workspace_b = nullptr;
    KOKKOS_HIP_CHECK(hipMalloc((void **)&workspace_b, sizeof(I) * matrix_a.nnz))
    (spgemm_numeric_block_per_row_sort_multipass_kernel<T, I, CHUNK_SIZE, BLOCK_THREAD_NUM, VEC_SIZE,
                                                        WF_SIZE>)<<<matrix_c.rows, 512>>>(
        alpha, matrix_c.rows, matrix_c.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
        matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, matrix_c.row_ptr, matrix_c.col_idx_ptr,
        matrix_c.values_ptr, workspace_b);
    is_sorted = true;
    // HashMap<T, I> hash_map(matrix_c.col_idx_ptr, matrix_c.values_ptr, matrix_c.rows, matrix_c.nnz);
    // if (matrix_c.cols < 32 * HashMap<T, I>::Capacity) {
    //   using BitSetEleType = unsigned int;
    //   const size_t idx_bits = min(max(matrix_c.cols / HashMap<T, I>::Capacity, 0) + 1, sizeof(BitSetEleType) * 8);
    //   constexpr size_t bits_num_per_key = 1;

    //   I *tmp_col_idx_ptr = nullptr;
    //   T *tmp_values_ptr = nullptr;
    //   KOKKOS_HIP_CHECK(hipMalloc((void **)&tmp_col_idx_ptr, matrix_c.nnz * sizeof(I)))
    //   KOKKOS_HIP_CHECK(hipMalloc((void **)&tmp_values_ptr, matrix_c.nnz * sizeof(T)))

    //   constexpr size_t LDS_MAP_CAP = 1 << 5;
    //   constexpr size_t LDS_HASH_MASK = LDS_MAP_CAP - 1;
    //   constexpr size_t GLOBAL_BLOCK_NUM = 512;
    //   constexpr size_t BLOCK_THREAD_NUM = 256;
    //   constexpr size_t N = 4;

    //   (spgemm_numeric_block_per_row_sort_kernel<T, I, BitSetEleType, N, LDS_MAP_CAP, LDS_HASH_MASK, BLOCK_THREAD_NUM,
    //                                             WF_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
    //       alpha, matrix_a.rows, matrix_a.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
    //       matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, matrix_c.row_ptr, hash_map.begins,
    //       hash_map.nexts, idx_bits, hash_map.ids, hash_map.values, tmp_col_idx_ptr, tmp_values_ptr);

    //   KOKKOS_HIP_CHECK(hipFree(matrix_c.col_idx_ptr));
    //   KOKKOS_HIP_CHECK(hipFree(matrix_c.values_ptr));

    //   matrix_c.col_idx_ptr = tmp_col_idx_ptr;
    //   matrix_c.values_ptr = tmp_values_ptr;
    //   hash_map.ids = tmp_col_idx_ptr;
    //   hash_map.values = tmp_values_ptr;
    //   is_sorted = true;
    // } else {
    //   constexpr size_t LDS_MAP_CAP = 1 << 5;
    //   constexpr size_t LDS_HASH_MASK = LDS_MAP_CAP - 1;
    //   constexpr size_t GLOBAL_BLOCK_NUM = 512;
    //   constexpr size_t BLOCK_THREAD_NUM = 256;
    //   constexpr size_t N = 4;

    //   (spgemm_numeric_block_per_row_kernel<T, I, N, LDS_MAP_CAP, LDS_HASH_MASK, BLOCK_THREAD_NUM,
    //                                        WF_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
    //       alpha, matrix_a.rows, matrix_a.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
    //       matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, matrix_c.row_ptr, hash_map.begins,
    //       hash_map.nexts, hash_map.ids, hash_map.values);
    //   is_sorted = false;
    // }
  }
}

#endif // SPGEMM_KOKKOS_SPGEMM_NUMERIC_IMPL_H
