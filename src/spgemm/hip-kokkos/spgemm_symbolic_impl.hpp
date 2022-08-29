#ifndef SPGEMM_KOKKOS_SPGEMM_SYMBOLIC_IMPL_HPP
#define SPGEMM_KOKKOS_SPGEMM_SYMBOLIC_IMPL_HPP

#include "compress_matrix.h"
#include "csr_matrix.h"
#include "error_macro.hpp"
#include "max_num.h"
#include "scan.h"
#include "spgemm_compress.h"
#include "spgemm_symbolic_kernel.hpp"
#include <cstddef>
#include <hip/hip_runtime.h>
#ifdef DEBUG_MODE
#include <iostream>
#endif

template <typename T, typename I>
void spgemm_symbolic(const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b, I *row_ptr_c, size_t *nnz_c,
                     I *max_row_num, I *temp_buffer, const size_t temp_buffer_size) {
  // compress matrix B
  CompressMatrix<I> compress_matrix_b = spgemm_compress<T, I>(matrix_b);
  constexpr size_t WF_SIZE = 64;
  // symbolic
  // max num in bits
  I max_bits_num =
      matrix_b.cols / CompressMatrix<I>::BitsLen + (matrix_b.cols % CompressMatrix<I>::BitsLen == 0 ? 0 : 1);

  // buffer for scan
  I *scan_temp_buffer = nullptr;
  const size_t scan_temp_buffer_size = scan_in_place_buffer_size<I>(matrix_a.rows + 1);
  if (temp_buffer_size >= scan_temp_buffer_size) {
    scan_temp_buffer = reinterpret_cast<I *>(temp_buffer);
  } else {
    KOKKOS_HIP_CHECK(hipMalloc((void **)&scan_temp_buffer, scan_temp_buffer_size))
  }
  // buffer for max
  I *max_temp_buffer = nullptr;

  const size_t max_temp_buffer_size = max_num_buffer_size<I>(matrix_a.rows + 1);
  if (temp_buffer_size >= max_temp_buffer_size) {
    max_temp_buffer = reinterpret_cast<I *>(temp_buffer);
  } else {
    KOKKOS_HIP_CHECK(hipMalloc((void **)&max_temp_buffer, max_temp_buffer_size))
  }

#ifdef DEBUG_MODE
  std::cout << "max_bits_num: " << max_bits_num << std::endl;
#endif

#ifdef SYMBOLIC_LINEAR_CHAIN_HASHMAP
#define SYMBOLIC_VECTOR_PER_ROW_ONE_PASS_METHOD                                                                        \
  spgemm_symbolic_linear_chain_hashmap_vector_per_row_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM, VEC_SIZE, WF_SIZE>
#define SYMBOLIC_BLOCK_PER_ROW_ONE_PASS_METHOD                                                                         \
  spgemm_symbolic_linear_chain_hashmap_block_per_row_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM, WF_SIZE>
#endif

#ifndef SYMBOLIC_LINEAR_CHAIN_HASHMAP
#define SYMBOLIC_VECTOR_PER_ROW_ONE_PASS_METHOD                                                                        \
  spgemm_symbolic_open_address_hashmap_vector_per_row_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM, VEC_SIZE, WF_SIZE>
#define SYMBOLIC_BLOCK_PER_ROW_ONE_PASS_METHOD                                                                         \
  spgemm_symbolic_open_address_hashmap_block_per_row_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM, WF_SIZE>
#endif

  if (max_bits_num < 16) {
    constexpr size_t N = 2;
    constexpr size_t VEC_SIZE = 16 / N;
    constexpr size_t BLOCK_THREAD_NUM = 256;
    constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
    const size_t GLOBAL_BLOCK_NUM = matrix_a.rows / BLOCK_VEC_NUM + (matrix_a.rows % BLOCK_VEC_NUM == 0 ? 0 : 1);

    (spgemm_symbolic_vector_per_row_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM, VEC_SIZE,
                                                    WF_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
        matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr, compress_matrix_b.ids_ptr,
        compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_ptr_c);
  } else if (max_bits_num < 32) {
    constexpr size_t N = 2;
    constexpr size_t VEC_SIZE = 32 / N;
    constexpr size_t BLOCK_THREAD_NUM = 256;
    constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
    const size_t GLOBAL_BLOCK_NUM = matrix_a.rows / BLOCK_VEC_NUM + (matrix_a.rows % BLOCK_VEC_NUM == 0 ? 0 : 1);

    (spgemm_symbolic_vector_per_row_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM, VEC_SIZE,
                                                    WF_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
        matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr, compress_matrix_b.ids_ptr,
        compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_ptr_c);
  } else if (max_bits_num < 64) {
    constexpr size_t N = 2;
    constexpr size_t VEC_SIZE = 64 / N;
    constexpr size_t BLOCK_THREAD_NUM = 256;
    constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
    const size_t GLOBAL_BLOCK_NUM = matrix_a.rows / BLOCK_VEC_NUM + (matrix_a.rows % BLOCK_VEC_NUM == 0 ? 0 : 1);

    (spgemm_symbolic_vector_per_row_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM, VEC_SIZE,
                                                    WF_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
        matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr, compress_matrix_b.ids_ptr,
        compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_ptr_c);
  } else if (max_bits_num < 128) {
    constexpr size_t N = 2;
    constexpr size_t VEC_SIZE = 128 / N;
    constexpr size_t BLOCK_THREAD_NUM = 256;
    constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
    const size_t GLOBAL_BLOCK_NUM = matrix_a.rows / BLOCK_VEC_NUM + (matrix_a.rows % BLOCK_VEC_NUM == 0 ? 0 : 1);

    (spgemm_symbolic_vector_per_row_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM, VEC_SIZE,
                                                    WF_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
        matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr, compress_matrix_b.ids_ptr,
        compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_ptr_c);
  } else if (max_bits_num < 512) {
    constexpr size_t N = 2;
    constexpr size_t BLOCK_THREAD_NUM = 512 / N;
    const size_t GLOBAL_BLOCK_NUM = matrix_a.rows;

    (spgemm_symbolic_block_per_row_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM,
                                                   WF_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
        matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr, compress_matrix_b.ids_ptr,
        compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_ptr_c);
  } else if (max_bits_num < 1024) {
    constexpr size_t N = 2;
    constexpr size_t BLOCK_THREAD_NUM = 1024 / N;
    const size_t GLOBAL_BLOCK_NUM = matrix_a.rows;

    (spgemm_symbolic_block_per_row_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM,
                                                   WF_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
        matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr, compress_matrix_b.ids_ptr,
        compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_ptr_c);
  } else {
    // calc size of trunk
    I *dchunk_num_ptr = nullptr;
    KOKKOS_HIP_CHECK(hipMalloc((void **)(&dchunk_num_ptr), (matrix_a.rows + 1) * sizeof(I)))
    KOKKOS_HIP_CHECK(hipMemset(dchunk_num_ptr, 0, sizeof(I)))
    (trunk_nums_kernel<I, 32, 64>)<<<512, 256>>>(matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr,
                                                 compress_matrix_b.use_count, dchunk_num_ptr);
    I max_chunk_num = static_cast<I>(0);
    // max num in row_ptr
    I *dmax_chunk_num_ptr = nullptr;
    KOKKOS_HIP_CHECK(hipMalloc((void **)(&dmax_chunk_num_ptr), sizeof(I)));
    max_num_in_array<I>(matrix_a.rows + 1, dchunk_num_ptr, dmax_chunk_num_ptr, max_temp_buffer);
    KOKKOS_HIP_CHECK(hipMemcpy(&max_chunk_num, dmax_chunk_num_ptr, sizeof(I), hipMemcpyDeviceToHost))
    KOKKOS_HIP_CHECK(hipFree(dmax_chunk_num_ptr))
#ifdef DEBUG_MODE
    std::cout << "max_chunk_num: " << max_chunk_num << std::endl;
#endif
    max_chunk_num = matrix_b.cols < max_chunk_num ? matrix_b.cols : max_chunk_num;
    if (max_chunk_num < 16) {
      constexpr size_t N = 2;
      constexpr size_t VEC_SIZE = 16 / N;
      constexpr size_t BLOCK_THREAD_NUM = 256;
      constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
      const size_t GLOBAL_BLOCK_NUM = matrix_a.rows / BLOCK_VEC_NUM + (matrix_a.rows % BLOCK_VEC_NUM == 0 ? 0 : 1);

      (SYMBOLIC_VECTOR_PER_ROW_ONE_PASS_METHOD)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr, compress_matrix_b.ids_ptr,
          compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_ptr_c);
    } else if (max_chunk_num < 32) {
      constexpr size_t N = 2;
      constexpr size_t VEC_SIZE = 32 / N;
      constexpr size_t BLOCK_THREAD_NUM = 256;
      constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
      const size_t GLOBAL_BLOCK_NUM = matrix_a.rows / BLOCK_VEC_NUM + (matrix_a.rows % BLOCK_VEC_NUM == 0 ? 0 : 1);

      (SYMBOLIC_VECTOR_PER_ROW_ONE_PASS_METHOD)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr, compress_matrix_b.ids_ptr,
          compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_ptr_c);

    } else if (max_chunk_num < 64) {
      constexpr size_t N = 2;
      constexpr size_t VEC_SIZE = 64 / N;
      constexpr size_t BLOCK_THREAD_NUM = 256;
      constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
      const size_t GLOBAL_BLOCK_NUM = matrix_a.rows / BLOCK_VEC_NUM + (matrix_a.rows % BLOCK_VEC_NUM == 0 ? 0 : 1);

      (SYMBOLIC_VECTOR_PER_ROW_ONE_PASS_METHOD)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr, compress_matrix_b.ids_ptr,
          compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_ptr_c);

    } else if (max_chunk_num < 128) {
      constexpr size_t N = 2;
      constexpr size_t VEC_SIZE = 128 / N;
      constexpr size_t BLOCK_THREAD_NUM = 256;
      constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
      const size_t GLOBAL_BLOCK_NUM = matrix_a.rows / BLOCK_VEC_NUM + (matrix_a.rows % BLOCK_VEC_NUM == 0 ? 0 : 1);

      (SYMBOLIC_VECTOR_PER_ROW_ONE_PASS_METHOD)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr, compress_matrix_b.ids_ptr,
          compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_ptr_c);

    } else if (max_chunk_num < 512) {
      constexpr size_t N = 2;
      constexpr size_t BLOCK_THREAD_NUM = 512 / N;
      const size_t GLOBAL_BLOCK_NUM = matrix_a.rows;
      (SYMBOLIC_BLOCK_PER_ROW_ONE_PASS_METHOD)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr, compress_matrix_b.ids_ptr,
          compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_ptr_c);
    } else if (max_chunk_num < 1024) {
      constexpr size_t N = 2;
      constexpr size_t BLOCK_THREAD_NUM = 1024 / N;
      const size_t GLOBAL_BLOCK_NUM = matrix_a.rows;
      (SYMBOLIC_BLOCK_PER_ROW_ONE_PASS_METHOD)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr, compress_matrix_b.ids_ptr,
          compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_ptr_c);
    } else if (max_chunk_num < 2048) {
      constexpr size_t N = 2;
      constexpr size_t BLOCK_THREAD_NUM = 2048 / N;
      const size_t GLOBAL_BLOCK_NUM = matrix_a.rows;
      (SYMBOLIC_BLOCK_PER_ROW_ONE_PASS_METHOD)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr, compress_matrix_b.ids_ptr,
          compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_ptr_c);
    } else {
      using idType = typename CompressMatrix<I>::IdType;
      using bitType = typename CompressMatrix<I>::BitSetEleType;
      scan_in_place<I>(matrix_a.rows + 1, dchunk_num_ptr, scan_temp_buffer);

      I chunk_num_c = 0;
      KOKKOS_HIP_CHECK(hipMemcpy(&chunk_num_c, dchunk_num_ptr + matrix_a.rows, sizeof(I), hipMemcpyDeviceToHost))
      // alloc hashmap
      idType *map_ids = nullptr;
      bitType *map_values = nullptr;
      int *map_use_count = nullptr;
      KOKKOS_HIP_CHECK(hipMalloc((void **)(&map_ids), chunk_num_c * sizeof(idType)));
      KOKKOS_HIP_CHECK(hipMalloc((void **)(&map_values), chunk_num_c * sizeof(bitType)));
      KOKKOS_HIP_CHECK(hipMalloc((void **)(&map_use_count), matrix_a.rows * sizeof(int)));

      HashMap<bitType, idType> hash_map(map_ids, map_values, matrix_a.rows, chunk_num_c);
      constexpr size_t LDS_MAP_CAP = 1 << 5;
      constexpr size_t LDS_HASH_MASK = LDS_MAP_CAP - 1;
      const size_t GLOBAL_BLOCK_NUM = 128;
      constexpr size_t BLOCK_THREAD_NUM = 128;
      constexpr size_t VEC_SIZE = 32;
      constexpr size_t BLOCK_WF_NUM = BLOCK_THREAD_NUM / WF_SIZE;
      constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
      constexpr size_t N = 4;
      (spgemm_symbolic_linear_chain_hashmap_block_per_row_kernel<T, I, N, LDS_MAP_CAP, LDS_HASH_MASK, BLOCK_THREAD_NUM,
                                                                 WF_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr, compress_matrix_b.ids_ptr,
          compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, dchunk_num_ptr, hash_map.begins, hash_map.nexts,
          hash_map.ids, hash_map.values, map_use_count);
      // calc row_ptr
      (row_nnz_kernel<I, bitType, 32, 64>
       <<<512, 256>>>(matrix_a.rows, dchunk_num_ptr, hash_map.values, map_use_count, row_ptr_c));

      // free memory
      KOKKOS_HIP_CHECK(hipFree(map_use_count))
      KOKKOS_HIP_CHECK(hipFree(map_values))
      KOKKOS_HIP_CHECK(hipFree(map_ids))
    }
    KOKKOS_HIP_CHECK(hipFree(dchunk_num_ptr))
  }

  // max num in row_ptr
  I *device_max_row_num = nullptr;
  KOKKOS_HIP_CHECK(hipMalloc((void **)(&device_max_row_num), sizeof(I)));
  max_num_in_array<I>(matrix_a.rows + 1, row_ptr_c, device_max_row_num, max_temp_buffer);
  KOKKOS_HIP_CHECK(hipMemcpy(max_row_num, device_max_row_num, sizeof(I), hipMemcpyDeviceToHost))
  KOKKOS_HIP_CHECK(hipFree(device_max_row_num))
  // reduce than fan
  scan_in_place<I>(matrix_a.rows + 1, row_ptr_c, scan_temp_buffer);

  // set nnz of matrix c
  KOKKOS_HIP_CHECK(hipMemcpy(nnz_c, row_ptr_c + matrix_a.rows, sizeof(I), hipMemcpyDeviceToHost))

  if (temp_buffer_size < scan_temp_buffer_size) {
    KOKKOS_HIP_CHECK(hipFree(max_temp_buffer))
  }
  if (temp_buffer_size < max_temp_buffer_size) {
    KOKKOS_HIP_CHECK(hipFree(scan_temp_buffer))
  }
}

#endif // SPGEMM_KOKKOS_SPGEMM_SYMBOLIC_IMPL_HPP
