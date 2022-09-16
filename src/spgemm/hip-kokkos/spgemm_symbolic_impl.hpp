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

template <typename T, typename I>
size_t spgemm_symbolic_compress_buffer_size(const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b) {
  size_t buffer_size = 0;
  size_t alloc_size = 0;
  alloc_size = spgemm_compress_buffer_size(matrix_b);
  buffer_size += ALIGN_256_BYTES(alloc_size);

  alloc_size = common::scan_buffer_size<I>(matrix_a.rows + 1);
  buffer_size += ALIGN_256_BYTES(alloc_size);

  alloc_size = common::max_buffer_size<I>(matrix_a.rows + 1);
  buffer_size += ALIGN_256_BYTES(alloc_size);

  alloc_size = (matrix_a.rows + 1) * sizeof(I);
  buffer_size += ALIGN_256_BYTES(alloc_size);

  alloc_size = sizeof(I);
  buffer_size += ALIGN_256_BYTES(alloc_size);

  alloc_size = sizeof(I);
  buffer_size += ALIGN_256_BYTES(alloc_size);
  return buffer_size;
}

template <typename T, typename I>
void spgemm_symbolic_compress(const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b, I *row_ptr_c,
                              size_t *hnnz_c, I *hrow_nnz_max, void *temp_buffer) {
  char *ptr = reinterpret_cast<char *>(temp_buffer);
  size_t alloc_size = 0;
  size_t align_size = 0;
  // compress matrix B
  CompressMatrix<I> compress_matrix_b = spgemm_compress<T, I>(matrix_b, reinterpret_cast<void *>(ptr));

  alloc_size = spgemm_compress_buffer_size(matrix_b);
  align_size = ALIGN_256_BYTES(alloc_size);
  ptr += align_size;

  // max num in bits
  I max_bits_num =
      matrix_b.cols / CompressMatrix<I>::BitsLen + (matrix_b.cols % CompressMatrix<I>::BitsLen == 0 ? 0 : 1);
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

  if (max_bits_num <= 16) {
    constexpr size_t N = 2;
    constexpr size_t VEC_SIZE = 16 / N;
    constexpr size_t BLOCK_THREAD_NUM = 256;
    constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
    const size_t GLOBAL_BLOCK_NUM = matrix_a.rows / BLOCK_VEC_NUM + (matrix_a.rows % BLOCK_VEC_NUM == 0 ? 0 : 1);

    (spgemm_symbolic_vector_per_row_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM,
                                                    VEC_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
        matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr, compress_matrix_b.ids_ptr,
        compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_ptr_c);
  } else if (max_bits_num <= 32) {
    constexpr size_t N = 2;
    constexpr size_t VEC_SIZE = 32 / N;
    constexpr size_t BLOCK_THREAD_NUM = 256;
    constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
    const size_t GLOBAL_BLOCK_NUM = matrix_a.rows / BLOCK_VEC_NUM + (matrix_a.rows % BLOCK_VEC_NUM == 0 ? 0 : 1);

    (spgemm_symbolic_vector_per_row_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM,
                                                    VEC_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
        matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr, compress_matrix_b.ids_ptr,
        compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_ptr_c);
  } else if (max_bits_num <= 64) {
    constexpr size_t N = 2;
    constexpr size_t VEC_SIZE = 64 / N;
    constexpr size_t BLOCK_THREAD_NUM = 256;
    constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
    const size_t GLOBAL_BLOCK_NUM = matrix_a.rows / BLOCK_VEC_NUM + (matrix_a.rows % BLOCK_VEC_NUM == 0 ? 0 : 1);

    (spgemm_symbolic_vector_per_row_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM,
                                                    VEC_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
        matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr, compress_matrix_b.ids_ptr,
        compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_ptr_c);
  } else if (max_bits_num <= 128) {
    constexpr size_t N = 2;
    constexpr size_t VEC_SIZE = 128 / N;
    constexpr size_t BLOCK_THREAD_NUM = 256;
    constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
    const size_t GLOBAL_BLOCK_NUM = matrix_a.rows / BLOCK_VEC_NUM + (matrix_a.rows % BLOCK_VEC_NUM == 0 ? 0 : 1);

    (spgemm_symbolic_vector_per_row_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM,
                                                    VEC_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
        matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr, compress_matrix_b.ids_ptr,
        compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_ptr_c);
  } else if (max_bits_num <= 512) {
    constexpr size_t N = 2;
    constexpr size_t BLOCK_THREAD_NUM = 512 / N;
    const size_t GLOBAL_BLOCK_NUM = matrix_a.rows;

    (spgemm_symbolic_block_per_row_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
        matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr, compress_matrix_b.ids_ptr,
        compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_ptr_c);
  } else if (max_bits_num <= 1024) {
    constexpr size_t N = 2;
    constexpr size_t BLOCK_THREAD_NUM = 1024 / N;
    const size_t GLOBAL_BLOCK_NUM = matrix_a.rows;

    (spgemm_symbolic_block_per_row_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
        matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr, compress_matrix_b.ids_ptr,
        compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_ptr_c);
  } else if (max_bits_num <= 2048) {
    constexpr size_t N = 2;
    constexpr size_t BLOCK_THREAD_NUM = 2048 / N;
    const size_t GLOBAL_BLOCK_NUM = matrix_a.rows;

    (spgemm_symbolic_block_per_row_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
        matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr, compress_matrix_b.ids_ptr,
        compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_ptr_c);
  } else {
    // calc size of trunk
    I *chunk_num_ptr = reinterpret_cast<I *>(ptr);
    alloc_size = (matrix_a.rows + 1) * sizeof(I);
    align_size = ALIGN_256_BYTES(alloc_size);
    ptr += align_size;
    (trunk_nums_compress_kernel<I, 4>)<<<512, 256>>>(matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr,
                                                     compress_matrix_b.use_count, chunk_num_ptr);

    // max of trunk num
    I hchunk_num_max = static_cast<I>(0);
    {
      I *chunk_num_max_ptr = reinterpret_cast<I *>(ptr);
      alloc_size = sizeof(I);
      align_size = ALIGN_256_BYTES(alloc_size);
      ptr += align_size;
      common::max<I>(matrix_a.rows + 1, chunk_num_ptr, chunk_num_max_ptr, max_temp_buffer);
      HIP_CHECK(hipMemcpy(&hchunk_num_max, chunk_num_max_ptr, sizeof(I), hipMemcpyDeviceToHost))
    }

    hchunk_num_max = std::min<I>(matrix_b.cols, hchunk_num_max);

    if (hchunk_num_max <= 16) {
      constexpr size_t N = 2;
      constexpr size_t VEC_SIZE = 16 / N;
      constexpr size_t BLOCK_THREAD_NUM = 256;
      constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
      const size_t GLOBAL_BLOCK_NUM = matrix_a.rows / BLOCK_VEC_NUM + (matrix_a.rows % BLOCK_VEC_NUM == 0 ? 0 : 1);

      spgemm_symbolic_open_address_hashmap_vector_per_row_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM, VEC_SIZE>
          <<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
              matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr,
              compress_matrix_b.ids_ptr, compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_ptr_c);

    } else if (hchunk_num_max <= 32) {
      constexpr size_t N = 2;
      constexpr size_t VEC_SIZE = 32 / N;
      constexpr size_t BLOCK_THREAD_NUM = 256;
      constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
      const size_t GLOBAL_BLOCK_NUM = matrix_a.rows / BLOCK_VEC_NUM + (matrix_a.rows % BLOCK_VEC_NUM == 0 ? 0 : 1);

      spgemm_symbolic_open_address_hashmap_vector_per_row_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM, VEC_SIZE>
          <<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
              matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr,
              compress_matrix_b.ids_ptr, compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_ptr_c);

    } else if (hchunk_num_max <= 64) {
      constexpr size_t N = 2;
      constexpr size_t VEC_SIZE = 64 / N;
      constexpr size_t BLOCK_THREAD_NUM = 256;
      constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
      const size_t GLOBAL_BLOCK_NUM = matrix_a.rows / BLOCK_VEC_NUM + (matrix_a.rows % BLOCK_VEC_NUM == 0 ? 0 : 1);

      spgemm_symbolic_open_address_hashmap_vector_per_row_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM, VEC_SIZE>
          <<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
              matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr,
              compress_matrix_b.ids_ptr, compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_ptr_c);

    } else if (hchunk_num_max <= 128) {
      constexpr size_t N = 2;
      constexpr size_t VEC_SIZE = 128 / N;
      constexpr size_t BLOCK_THREAD_NUM = 256;
      constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
      const size_t GLOBAL_BLOCK_NUM = matrix_a.rows / BLOCK_VEC_NUM + (matrix_a.rows % BLOCK_VEC_NUM == 0 ? 0 : 1);

      spgemm_symbolic_open_address_hashmap_vector_per_row_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM, VEC_SIZE>
          <<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
              matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr,
              compress_matrix_b.ids_ptr, compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_ptr_c);

    } else if (hchunk_num_max <= 512) {
      constexpr size_t N = 2;
      constexpr size_t BLOCK_THREAD_NUM = 512 / N;
      const size_t GLOBAL_BLOCK_NUM = matrix_a.rows;
      spgemm_symbolic_open_address_hashmap_block_per_row_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM>
          <<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
              matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr,
              compress_matrix_b.ids_ptr, compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_ptr_c);

    } else if (hchunk_num_max <= 1024) {
      constexpr size_t N = 2;
      constexpr size_t BLOCK_THREAD_NUM = 1024 / N;
      const size_t GLOBAL_BLOCK_NUM = matrix_a.rows;
      spgemm_symbolic_open_address_hashmap_block_per_row_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM>
          <<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
              matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr,
              compress_matrix_b.ids_ptr, compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_ptr_c);

    } else if (hchunk_num_max <= 2048) {
      constexpr size_t N = 2;
      constexpr size_t BLOCK_THREAD_NUM = 2048 / N;
      const size_t GLOBAL_BLOCK_NUM = matrix_a.rows;
      spgemm_symbolic_open_address_hashmap_block_per_row_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM>
          <<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
              matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr,
              compress_matrix_b.ids_ptr, compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_ptr_c);

    } else {
      // todo: impl multipass kernel
      using idType = typename CompressMatrix<I>::IdType;
      using bitType = typename CompressMatrix<I>::BitSetEleType;
      common::scan<I>(matrix_a.rows + 1, chunk_num_ptr, scan_temp_buffer);

      I chunk_num_c = 0;
      HIP_CHECK(hipMemcpy(&chunk_num_c, chunk_num_ptr + matrix_a.rows, sizeof(I), hipMemcpyDeviceToHost))
      // alloc hashmap
      idType *map_ids = nullptr;
      bitType *map_values = nullptr;
      int *map_use_count = nullptr;
      HIP_CHECK(hipMalloc((void **)(&map_ids), chunk_num_c * sizeof(idType)));
      HIP_CHECK(hipMalloc((void **)(&map_values), chunk_num_c * sizeof(bitType)));
      HIP_CHECK(hipMalloc((void **)(&map_use_count), matrix_a.rows * sizeof(int)));

      HashMap<bitType, idType> hash_map(map_ids, map_values, matrix_a.rows, chunk_num_c);
      constexpr size_t LDS_MAP_CAP = 1 << 5;
      constexpr size_t LDS_HASH_MASK = LDS_MAP_CAP - 1;
      const size_t GLOBAL_BLOCK_NUM = 128;
      constexpr size_t BLOCK_THREAD_NUM = 128;
      constexpr size_t VEC_SIZE = 32;
      constexpr size_t BLOCK_WF_NUM = BLOCK_THREAD_NUM / __WF_SIZE__;
      constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
      constexpr size_t N = 4;
      (spgemm_symbolic_linear_chain_hashmap_block_per_row_kernel<
          T, I, N, LDS_MAP_CAP, LDS_HASH_MASK, BLOCK_THREAD_NUM>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr, compress_matrix_b.ids_ptr,
          compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, chunk_num_ptr, hash_map.begins, hash_map.nexts,
          hash_map.ids, hash_map.values, map_use_count);
      // calc row_ptr
      (row_nnz_kernel<I, bitType, 32>
       <<<512, 256>>>(matrix_a.rows, chunk_num_ptr, hash_map.values, map_use_count, row_ptr_c));

      // free memory
      HIP_CHECK(hipFree(map_use_count))
      HIP_CHECK(hipFree(map_values))
      HIP_CHECK(hipFree(map_ids))
    }
  }

  // max num in row_ptr
  {
    I *row_nnz_max_ptr = reinterpret_cast<I *>(ptr);
    const size_t alloc_size = sizeof(I);
    const size_t align_size = ALIGN_256_BYTES(alloc_size);
    ptr += align_size;
    common::max<I>(matrix_a.rows + 1, row_ptr_c, row_nnz_max_ptr, max_temp_buffer);
    HIP_CHECK(hipMemcpy(hrow_nnz_max, row_nnz_max_ptr, sizeof(I), hipMemcpyDeviceToHost))
  }
  // scan row_ptr
  common::scan<I>(matrix_a.rows + 1, row_ptr_c, scan_temp_buffer);
  // set nnz of matrix c
  HIP_CHECK(hipMemcpy(hnnz_c, row_ptr_c + matrix_a.rows, sizeof(I), hipMemcpyDeviceToHost))
}

template <typename T, typename I>
size_t spgemm_symbolic_buffer_size(const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b) {
  size_t buffer_size = 0;
  size_t alloc_size = 0;

  alloc_size = common::scan_buffer_size<I>(matrix_a.rows + 1);
  buffer_size += ALIGN_256_BYTES(alloc_size);

  alloc_size = common::max_buffer_size<I>(matrix_a.rows + 1);
  buffer_size += ALIGN_256_BYTES(alloc_size);

  alloc_size = (matrix_a.rows + 1) * sizeof(I);
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
  char *ptr = reinterpret_cast<char *>(temp_buffer);
  size_t alloc_size = 0;
  size_t align_size = 0;

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

  // calc size of trunk
  I *chunk_num_ptr = reinterpret_cast<I *>(ptr);
  alloc_size = (matrix_a.rows + 1) * sizeof(I);
  align_size = ALIGN_256_BYTES(alloc_size);
  ptr += align_size;
  (trunk_nums_kernel<I, 8>)<<<512, 256>>>(matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_b.row_ptr,
                                          chunk_num_ptr);

  // max of trunk num
  I hchunk_num_max = static_cast<I>(0);
  {
    I *chunk_num_max_ptr = reinterpret_cast<I *>(ptr);
    alloc_size = sizeof(I);
    align_size = ALIGN_256_BYTES(alloc_size);
    ptr += align_size;
    common::max<I>(matrix_a.rows + 1, chunk_num_ptr, chunk_num_max_ptr, max_temp_buffer);
    HIP_CHECK(hipMemcpy(&hchunk_num_max, chunk_num_max_ptr, sizeof(I), hipMemcpyDeviceToHost))
  }

  hchunk_num_max = std::min<I>(matrix_b.cols, hchunk_num_max);
  if (hchunk_num_max <= 16) {
    constexpr size_t N = 2;
    constexpr size_t VEC_SIZE = 16 / N;
    constexpr size_t BLOCK_THREAD_NUM = 256;
    constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
    const size_t GLOBAL_BLOCK_NUM = matrix_a.rows / BLOCK_VEC_NUM + (matrix_a.rows % BLOCK_VEC_NUM == 0 ? 0 : 1);

    (spgemm_symbolic_open_address_hashmap_vector_per_row_one_pass_kernel<
        T, I, N, BLOCK_THREAD_NUM, VEC_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
        matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_b.row_ptr, matrix_b.col_idx_ptr, row_ptr_c);

  } else if (hchunk_num_max <= 32) {
    constexpr size_t N = 2;
    constexpr size_t VEC_SIZE = 32 / N;
    constexpr size_t BLOCK_THREAD_NUM = 256;
    constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
    const size_t GLOBAL_BLOCK_NUM = matrix_a.rows / BLOCK_VEC_NUM + (matrix_a.rows % BLOCK_VEC_NUM == 0 ? 0 : 1);

    (spgemm_symbolic_open_address_hashmap_vector_per_row_one_pass_kernel<
        T, I, N, BLOCK_THREAD_NUM, VEC_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
        matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_b.row_ptr, matrix_b.col_idx_ptr, row_ptr_c);

  } else if (hchunk_num_max <= 64) {
    constexpr size_t N = 2;
    constexpr size_t VEC_SIZE = 64 / N;
    constexpr size_t BLOCK_THREAD_NUM = 256;
    constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
    const size_t GLOBAL_BLOCK_NUM = matrix_a.rows / BLOCK_VEC_NUM + (matrix_a.rows % BLOCK_VEC_NUM == 0 ? 0 : 1);

    (spgemm_symbolic_open_address_hashmap_vector_per_row_one_pass_kernel<
        T, I, N, BLOCK_THREAD_NUM, VEC_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
        matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_b.row_ptr, matrix_b.col_idx_ptr, row_ptr_c);

  } else if (hchunk_num_max <= 128) {
    constexpr size_t N = 2;
    constexpr size_t VEC_SIZE = 128 / N;
    constexpr size_t BLOCK_THREAD_NUM = 256;
    constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
    const size_t GLOBAL_BLOCK_NUM = matrix_a.rows / BLOCK_VEC_NUM + (matrix_a.rows % BLOCK_VEC_NUM == 0 ? 0 : 1);

    (spgemm_symbolic_open_address_hashmap_vector_per_row_one_pass_kernel<
        T, I, N, BLOCK_THREAD_NUM, VEC_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
        matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_b.row_ptr, matrix_b.col_idx_ptr, row_ptr_c);

  } else if (hchunk_num_max <= 512) {
    constexpr size_t N = 2;
    constexpr size_t BLOCK_THREAD_NUM = 512 / N;
    const size_t GLOBAL_BLOCK_NUM = matrix_a.rows;

    (spgemm_symbolic_open_address_hashmap_block_per_row_one_pass_kernel<
        T, I, N, BLOCK_THREAD_NUM>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
        matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_b.row_ptr, matrix_b.col_idx_ptr, row_ptr_c);

  } else if (hchunk_num_max <= 1024) {
    constexpr size_t N = 2;
    constexpr size_t BLOCK_THREAD_NUM = 1024 / N;
    const size_t GLOBAL_BLOCK_NUM = matrix_a.rows;

    (spgemm_symbolic_open_address_hashmap_block_per_row_one_pass_kernel<
        T, I, N, BLOCK_THREAD_NUM>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
        matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_b.row_ptr, matrix_b.col_idx_ptr, row_ptr_c);

  } else if (hchunk_num_max <= 2048) {
    constexpr size_t N = 2;
    constexpr size_t BLOCK_THREAD_NUM = 2048 / N;
    const size_t GLOBAL_BLOCK_NUM = matrix_a.rows;

    (spgemm_symbolic_open_address_hashmap_block_per_row_one_pass_kernel<
        T, I, N, BLOCK_THREAD_NUM>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
        matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_b.row_ptr, matrix_b.col_idx_ptr, row_ptr_c);

  } else {
    // todo: impl multipass method.
    std::cerr << "not support: hchunk_num more than 2048.\n"
              << "file: " << __FILE__ << ", function: " << __FUNCTION__ << ", line" << __LINE__ << std::endl;
    exit(-1);
  }

  // max num in row_ptr
  {
    I *row_nnz_max_ptr = reinterpret_cast<I *>(ptr);
    alloc_size = sizeof(I);
    align_size = ALIGN_256_BYTES(alloc_size);
    ptr += align_size;
    common::max<I>(matrix_a.rows + 1, row_ptr_c, row_nnz_max_ptr, max_temp_buffer);
    HIP_CHECK(hipMemcpy(hrow_nnz_max, row_nnz_max_ptr, sizeof(I), hipMemcpyDeviceToHost))
  }
  // scan row_ptr
  common::scan<I>(matrix_a.rows + 1, row_ptr_c, scan_temp_buffer);
  // set nnz of matrix c
  HIP_CHECK(hipMemcpy(hnnz_c, row_ptr_c + matrix_a.rows, sizeof(I), hipMemcpyDeviceToHost))
}

#endif // SPGEMM_KOKKOS_SPGEMM_SYMBOLIC_IMPL_HPP
