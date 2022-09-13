#ifndef SPGEMM_KOKKOS_SPGEMM_SYMBOLIC_IMPL_HPP
#define SPGEMM_KOKKOS_SPGEMM_SYMBOLIC_IMPL_HPP

#include "../common/check/hip_check.hpp"
#include "../common/max/max.h"
#include "../common/scan/scan.h"
#include "align.h"
#include "building_config.h"
#include "compress_matrix.h"
#include "csr_matrix.h"
#include "spgemm_compress.h"
#include "spgemm_symbolic_kernel.hpp"
#include <algorithm>
#include <cstddef>
#include <hip/hip_runtime.h>
#ifdef DEBUG_MODE
#include <iostream>
#endif

template <typename T, typename I>
void spgemm_symbolic_compress(const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b, I *row_ptr_c,
                              size_t *hnnz_c, I *hrow_nnz_max, void *temp_buffer, const size_t temp_buffer_size) {
  // compress matrix B
  CompressMatrix<I> compress_matrix_b = spgemm_compress<T, I>(matrix_b);
  // max num in bits
  I max_bits_num =
      matrix_b.cols / CompressMatrix<I>::BitsLen + (matrix_b.cols % CompressMatrix<I>::BitsLen == 0 ? 0 : 1);

#ifdef DEBUG_MODE
  std::cout << "max_bits_num: " << max_bits_num << std::endl;
#endif

  bool is_scan_alloc = false;
  bool is_max_alloc = false;
  bool is_chunk_alloc = false;
  bool is_chunk_max_alloc = false;
  bool is_row_nnz_max_alloc = false;

  char *ptr = reinterpret_cast<char *>(temp_buffer);
  size_t remain_buffer_size = temp_buffer_size;
  // alloc buffer for scan op
  void *scan_temp_buffer = nullptr;
  {
    const size_t alloc_size = common::scan_buffer_size<I>(matrix_a.rows + 1);
    const size_t align_size = ALIGN_256_BYTES(alloc_size);
    if (remain_buffer_size >= align_size) {
      scan_temp_buffer = reinterpret_cast<void *>(ptr);
      ptr += align_size;
      remain_buffer_size -= align_size;
    } else {
      HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&scan_temp_buffer), alloc_size))
      is_scan_alloc = true;
    }
  }
  // alloc buffer for max op
  void *max_temp_buffer = nullptr;
  {
    const size_t alloc_size = common::max_buffer_size<I>(matrix_a.rows + 1);
    const size_t align_size = ALIGN_256_BYTES(alloc_size);
    if (remain_buffer_size >= align_size) {
      max_temp_buffer = reinterpret_cast<void *>(ptr);
      ptr += align_size;
      remain_buffer_size -= align_size;
    } else {
      HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&max_temp_buffer), alloc_size))
      is_max_alloc = true;
    }
  }

#ifdef SYMBOLIC_LINEAR_CHAIN_HASHMAP
#define SYMBOLIC_VECTOR_PER_ROW_ONE_PASS_METHOD                                                                        \
  spgemm_symbolic_linear_chain_hashmap_vector_per_row_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM, VEC_SIZE>
#define SYMBOLIC_BLOCK_PER_ROW_ONE_PASS_METHOD                                                                         \
  spgemm_symbolic_linear_chain_hashmap_block_per_row_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM>
#endif

#ifndef SYMBOLIC_LINEAR_CHAIN_HASHMAP
#define SYMBOLIC_VECTOR_PER_ROW_ONE_PASS_METHOD                                                                        \
  spgemm_symbolic_open_address_hashmap_vector_per_row_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM, VEC_SIZE>
#define SYMBOLIC_BLOCK_PER_ROW_ONE_PASS_METHOD                                                                         \
  spgemm_symbolic_open_address_hashmap_block_per_row_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM>
#endif

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
  } else {
    // calc size of trunk
    I *chunk_num_ptr = nullptr;
    {
      const size_t alloc_size = (matrix_a.rows + 1) * sizeof(I);
      const size_t align_size = ALIGN_256_BYTES(alloc_size);
      if (remain_buffer_size >= align_size) {
        chunk_num_ptr = reinterpret_cast<I *>(ptr);
        ptr += align_size;
        remain_buffer_size -= align_size;
      } else {
        HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&chunk_num_ptr), alloc_size))
        is_chunk_alloc = true;
      }
      HIP_CHECK(hipMemset(chunk_num_ptr, 0, sizeof(I)))
      (trunk_nums_compress_kernel<I, 64>)<<<512, 256>>>(matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr,
                                                        compress_matrix_b.use_count, chunk_num_ptr);
    }

    // max of trunk num
    I hchunk_num_max = static_cast<I>(0);
    {
      I *chunk_num_max_ptr = nullptr;
      const size_t alloc_size = sizeof(I);
      const size_t align_size = ALIGN_256_BYTES(alloc_size);
      if (remain_buffer_size >= align_size) {
        chunk_num_max_ptr = reinterpret_cast<I *>(ptr);
        ptr += align_size;
        remain_buffer_size -= align_size;
      } else {
        HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&chunk_num_max_ptr), alloc_size));
        is_chunk_max_alloc = true;
      }
      common::max<I>(matrix_a.rows + 1, chunk_num_ptr, chunk_num_max_ptr, max_temp_buffer);
      HIP_CHECK(hipMemcpy(&hchunk_num_max, chunk_num_max_ptr, sizeof(I), hipMemcpyDeviceToHost))

      if (is_chunk_max_alloc) {
        HIP_CHECK(hipFree(chunk_num_max_ptr))
      }
    }

    hchunk_num_max = std::min<I>(matrix_b.cols, hchunk_num_max);
#ifdef DEBUG_MODE
    std::cout << "hchunk_num_max: " << hchunk_num_max << std::endl;
#endif

    if (hchunk_num_max <= 16) {
      constexpr size_t N = 2;
      constexpr size_t VEC_SIZE = 16 / N;
      constexpr size_t BLOCK_THREAD_NUM = 256;
      constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
      const size_t GLOBAL_BLOCK_NUM = matrix_a.rows / BLOCK_VEC_NUM + (matrix_a.rows % BLOCK_VEC_NUM == 0 ? 0 : 1);

      (SYMBOLIC_VECTOR_PER_ROW_ONE_PASS_METHOD)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr, compress_matrix_b.ids_ptr,
          compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_ptr_c);

    } else if (hchunk_num_max <= 32) {
      constexpr size_t N = 2;
      constexpr size_t VEC_SIZE = 32 / N;
      constexpr size_t BLOCK_THREAD_NUM = 256;
      constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
      const size_t GLOBAL_BLOCK_NUM = matrix_a.rows / BLOCK_VEC_NUM + (matrix_a.rows % BLOCK_VEC_NUM == 0 ? 0 : 1);

      (SYMBOLIC_VECTOR_PER_ROW_ONE_PASS_METHOD)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr, compress_matrix_b.ids_ptr,
          compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_ptr_c);

    } else if (hchunk_num_max <= 64) {
      constexpr size_t N = 2;
      constexpr size_t VEC_SIZE = 64 / N;
      constexpr size_t BLOCK_THREAD_NUM = 256;
      constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
      const size_t GLOBAL_BLOCK_NUM = matrix_a.rows / BLOCK_VEC_NUM + (matrix_a.rows % BLOCK_VEC_NUM == 0 ? 0 : 1);

      (SYMBOLIC_VECTOR_PER_ROW_ONE_PASS_METHOD)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr, compress_matrix_b.ids_ptr,
          compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_ptr_c);

    } else if (hchunk_num_max <= 128) {
      constexpr size_t N = 2;
      constexpr size_t VEC_SIZE = 128 / N;
      constexpr size_t BLOCK_THREAD_NUM = 256;
      constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
      const size_t GLOBAL_BLOCK_NUM = matrix_a.rows / BLOCK_VEC_NUM + (matrix_a.rows % BLOCK_VEC_NUM == 0 ? 0 : 1);

      (SYMBOLIC_VECTOR_PER_ROW_ONE_PASS_METHOD)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr, compress_matrix_b.ids_ptr,
          compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_ptr_c);

    } else if (hchunk_num_max <= 512) {
      constexpr size_t N = 2;
      constexpr size_t BLOCK_THREAD_NUM = 512 / N;
      const size_t GLOBAL_BLOCK_NUM = matrix_a.rows;
      (SYMBOLIC_BLOCK_PER_ROW_ONE_PASS_METHOD)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr, compress_matrix_b.ids_ptr,
          compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_ptr_c);

    } else if (hchunk_num_max <= 1024) {
      constexpr size_t N = 2;
      constexpr size_t BLOCK_THREAD_NUM = 1024 / N;
      const size_t GLOBAL_BLOCK_NUM = matrix_a.rows;
      (SYMBOLIC_BLOCK_PER_ROW_ONE_PASS_METHOD)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr, compress_matrix_b.ids_ptr,
          compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_ptr_c);

    } else if (hchunk_num_max <= 2048) {
      constexpr size_t N = 2;
      constexpr size_t BLOCK_THREAD_NUM = 2048 / N;
      const size_t GLOBAL_BLOCK_NUM = matrix_a.rows;
      (SYMBOLIC_BLOCK_PER_ROW_ONE_PASS_METHOD)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr, compress_matrix_b.ids_ptr,
          compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_ptr_c);

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
    if (is_chunk_alloc) {
      HIP_CHECK(hipFree(chunk_num_ptr))
    }
  }

  // max num in row_ptr
  {
    I *row_nnz_max_ptr = nullptr;
    const size_t alloc_size = sizeof(I);
    const size_t align_size = ALIGN_256_BYTES(alloc_size);
    if (remain_buffer_size >= align_size) {
      row_nnz_max_ptr = reinterpret_cast<I *>(ptr);
      ptr += align_size;
      remain_buffer_size -= align_size;
    } else {
      HIP_CHECK(hipMalloc((void **)(&row_nnz_max_ptr), alloc_size));
      is_row_nnz_max_alloc = true;
    }
    common::max<I>(matrix_a.rows + 1, row_ptr_c, row_nnz_max_ptr, max_temp_buffer);
    HIP_CHECK(hipMemcpy(hrow_nnz_max, row_nnz_max_ptr, sizeof(I), hipMemcpyDeviceToHost))
    if (is_row_nnz_max_alloc) {
      HIP_CHECK(hipFree(row_nnz_max_ptr))
    }
  }
  // scan row_ptr
  common::scan<I>(matrix_a.rows + 1, row_ptr_c, scan_temp_buffer);
  // set nnz of matrix c
  HIP_CHECK(hipMemcpy(hnnz_c, row_ptr_c + matrix_a.rows, sizeof(I), hipMemcpyDeviceToHost))

  // free memory
  if (is_scan_alloc) {
    HIP_CHECK(hipFree(scan_temp_buffer))
  }
  if (is_max_alloc) {
    HIP_CHECK(hipFree(max_temp_buffer))
  }

#ifdef DEBUG_MODE
  std::cout << "alloc memory: \nscan\tmax\tchunk\tchunk-max\trow-nnz-max\t" << std::endl;
  std::cout << std::boolalpha << is_scan_alloc << "\t" << is_max_alloc << "\t" << is_chunk_alloc << "\t"
            << is_chunk_max_alloc << "\t" << is_row_nnz_max_alloc << std::endl;
#endif
}

template <typename T, typename I>
void spgemm_symbolic(const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b, I *row_ptr_c, size_t *hnnz_c,
                     I *hrow_nnz_max, void *temp_buffer, const size_t temp_buffer_size) {

  bool is_scan_alloc = false;
  bool is_max_alloc = false;
  bool is_chunk_alloc = false;
  bool is_chunk_max_alloc = false;
  bool is_row_nnz_max_alloc = false;

  char *ptr = reinterpret_cast<char *>(temp_buffer);
  size_t remain_buffer_size = temp_buffer_size;

  // alloc buffer for scan op
  void *scan_temp_buffer = nullptr;
  {
    const size_t alloc_size = common::scan_buffer_size<I>(matrix_a.rows + 1);
    const size_t align_size = ALIGN_256_BYTES(alloc_size);
    if (remain_buffer_size >= align_size) {
      scan_temp_buffer = reinterpret_cast<void *>(ptr);
      ptr += align_size;
      remain_buffer_size -= align_size;
    } else {
      HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&scan_temp_buffer), alloc_size))
      is_scan_alloc = true;
    }
  }
  // alloc buffer for max op
  void *max_temp_buffer = nullptr;
  {
    const size_t alloc_size = common::max_buffer_size<I>(matrix_a.rows + 1);
    const size_t align_size = ALIGN_256_BYTES(alloc_size);
    if (remain_buffer_size >= align_size) {
      max_temp_buffer = reinterpret_cast<void *>(ptr);
      ptr += align_size;
      remain_buffer_size -= align_size;
    } else {
      HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&max_temp_buffer), alloc_size))
      is_max_alloc = true;
    }
  }

  // calc size of trunk
  I *chunk_num_ptr = nullptr;
  {
    const size_t alloc_size = (matrix_a.rows + 1) * sizeof(I);
    const size_t align_size = ALIGN_256_BYTES(alloc_size);
    if (remain_buffer_size >= align_size) {
      chunk_num_ptr = reinterpret_cast<I *>(ptr);
      ptr += align_size;
      remain_buffer_size -= align_size;
    } else {
      HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&chunk_num_ptr), alloc_size))
      is_chunk_alloc = true;
    }
    HIP_CHECK(hipMemset(chunk_num_ptr, 0, sizeof(I)))
    (trunk_nums_kernel<I, 64>)<<<512, 256>>>(matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_b.row_ptr,
                                             chunk_num_ptr);
  }

  // max of trunk num
  I hchunk_num_max = static_cast<I>(0);
  {
    I *chunk_num_max_ptr = nullptr;
    const size_t alloc_size = sizeof(I);
    const size_t align_size = ALIGN_256_BYTES(alloc_size);
    if (remain_buffer_size >= align_size) {
      chunk_num_max_ptr = reinterpret_cast<I *>(ptr);
      ptr += align_size;
      remain_buffer_size -= align_size;
    } else {
      HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&chunk_num_max_ptr), alloc_size));
      is_chunk_max_alloc = true;
    }
    common::max<I>(matrix_a.rows + 1, chunk_num_ptr, chunk_num_max_ptr, max_temp_buffer);
    HIP_CHECK(hipMemcpy(&hchunk_num_max, chunk_num_max_ptr, sizeof(I), hipMemcpyDeviceToHost))

    if (is_chunk_max_alloc) {
      HIP_CHECK(hipFree(chunk_num_max_ptr))
    }
  }

  hchunk_num_max = std::min<I>(matrix_b.cols, hchunk_num_max);

#ifdef DEBUG_MODE
  std::cout << "hchunk_num_max: " << hchunk_num_max << std::endl;
#endif
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
    I *row_nnz_max_ptr = nullptr;
    const size_t alloc_size = sizeof(I);
    const size_t align_size = ALIGN_256_BYTES(alloc_size);
    if (remain_buffer_size >= align_size) {
      row_nnz_max_ptr = reinterpret_cast<I *>(ptr);
      ptr += align_size;
      remain_buffer_size -= align_size;
    } else {
      HIP_CHECK(hipMalloc((void **)(&row_nnz_max_ptr), alloc_size));
      is_row_nnz_max_alloc = true;
    }
    common::max<I>(matrix_a.rows + 1, row_ptr_c, row_nnz_max_ptr, max_temp_buffer);
    HIP_CHECK(hipMemcpy(hrow_nnz_max, row_nnz_max_ptr, sizeof(I), hipMemcpyDeviceToHost))
    if (is_row_nnz_max_alloc) {
      HIP_CHECK(hipFree(row_nnz_max_ptr))
    }
  }
  // scan row_ptr
  common::scan<I>(matrix_a.rows + 1, row_ptr_c, scan_temp_buffer);
  // set nnz of matrix c
  HIP_CHECK(hipMemcpy(hnnz_c, row_ptr_c + matrix_a.rows, sizeof(I), hipMemcpyDeviceToHost))

  // free memory
  if (is_scan_alloc) {
    HIP_CHECK(hipFree(scan_temp_buffer))
  }
  if (is_max_alloc) {
    HIP_CHECK(hipFree(max_temp_buffer))
  }

#ifdef DEBUG_MODE
  std::cout << "alloc memory: \nscan\tmax\tchunk\tchunk-max\trow-nnz-max\t" << std::endl;
  std::cout << std::boolalpha << is_scan_alloc << "\t" << is_max_alloc << "\t" << is_chunk_alloc << "\t"
            << is_chunk_max_alloc << "\t" << is_row_nnz_max_alloc << std::endl;
#endif
}

#endif // SPGEMM_KOKKOS_SPGEMM_SYMBOLIC_IMPL_HPP
