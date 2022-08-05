#ifndef SPGEMM_KOKKOS_SPGEMM_SYMBOLIC_IMPL_HPP
#define SPGEMM_KOKKOS_SPGEMM_SYMBOLIC_IMPL_HPP

#include "compress_matrix.h"
#include "csr_matrix.h"
#include "error_macro.hpp"
#include "max_num.h"
#include "scan.h"
#include "spgemm_symbolic_kernel.hpp"
#include <cstddef>
#include <hip/hip_runtime.h>

template <typename T, typename I>
void spgemm_symbolic(const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b, I *row_ptr_c, size_t *nnz_c,
                     I *max_row_num) {
  using idType = typename CompressMatrix<I>::IdType;
  using bitType = typename CompressMatrix<I>::BitSetEleType;

  // compress matrix B
  CompressMatrix<I> compress_matrix_b(matrix_b.rows, matrix_b.cols, matrix_b.nnz, matrix_b.row_ptr);
  (compress_matrix_kernel<I, CompressMatrix<I>::BitsLen, 32, 64>)<<<512, 256>>>(
      matrix_b.rows, matrix_b.row_ptr, matrix_b.col_idx_ptr, compress_matrix_b.ids_ptr, compress_matrix_b.col_idx_ptr,
      compress_matrix_b.use_count);

  // calc size of trunk
  I *row_trunk_ptr = nullptr;
  KOKKOS_HIP_CHECK(hipMalloc((void **)(&row_trunk_ptr), (matrix_a.rows + 1) * sizeof(I)))
  KOKKOS_HIP_CHECK(hipMemset(row_trunk_ptr, 0, sizeof(I)))
  (trunk_nums_kernel<I, 32, 64>)<<<512, 256>>>(matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr,
                                               compress_matrix_b.use_count, row_trunk_ptr);
  scan_in_place<I>(matrix_a.rows + 1, row_trunk_ptr);
  I trunk_num_c = 0;
  KOKKOS_HIP_CHECK(hipMemcpy(&trunk_num_c, row_trunk_ptr + matrix_a.rows, sizeof(I), hipMemcpyDeviceToHost))

  // alloc hashmap
  idType *map_ids = nullptr;
  bitType *map_values = nullptr;
  int *map_use_count = nullptr;
  KOKKOS_HIP_CHECK(hipMalloc((void **)(&map_ids), trunk_num_c * sizeof(idType)));
  KOKKOS_HIP_CHECK(hipMalloc((void **)(&map_values), trunk_num_c * sizeof(bitType)));
  KOKKOS_HIP_CHECK(hipMalloc((void **)(&map_use_count), matrix_a.rows * sizeof(int)));
  HashMap<bitType, idType> hash_map(map_ids, map_values, matrix_a.rows, trunk_num_c);

  // symbolic
  {
    constexpr size_t LDS_MAP_CAP = 1 << 5;
    constexpr size_t LDS_HASH_MASK = LDS_MAP_CAP - 1;
    constexpr size_t GLOBAL_BLOCK_NUM = 1024;
    constexpr size_t BLOCK_THREAD_NUM = 64;
    constexpr size_t VEC_SIZE = 32;
    constexpr size_t WF_SIZE = 64;
    constexpr size_t BLOCK_WF_NUM = BLOCK_THREAD_NUM / WF_SIZE;
    constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
    constexpr size_t N = 4;

    if (matrix_b.nnz / matrix_b.rows > 512) {
      constexpr size_t GLOBAL_BLOCK_NUM = 128;
      constexpr size_t BLOCK_THREAD_NUM = 128;
      constexpr size_t N = 1;
      (spgemm_symbolic_block_per_row_kernel<T, I, N, LDS_MAP_CAP, LDS_HASH_MASK, BLOCK_THREAD_NUM,
                                            WF_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr, compress_matrix_b.ids_ptr,
          compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_trunk_ptr, hash_map.begins, hash_map.nexts,
          hash_map.ids, hash_map.values, map_use_count);
    } else {
      (spgemm_symbolic_vector_per_row_kernel<T, I, N, LDS_MAP_CAP, LDS_HASH_MASK, BLOCK_THREAD_NUM, VEC_SIZE,
                                             WF_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr, compress_matrix_b.ids_ptr,
          compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_trunk_ptr, hash_map.begins, hash_map.nexts,
          hash_map.ids, hash_map.values, map_use_count);
    }
  }
  // calc row_ptr
  (row_nnz_kernel<I, bitType, 32, 64>
   <<<512, 256>>>(matrix_a.rows, row_trunk_ptr, hash_map.values, map_use_count, row_ptr_c));

  // max num in row_ptr
  I *device_max_row_num = nullptr;
  KOKKOS_HIP_CHECK(hipMalloc((void **)(&device_max_row_num), sizeof(I)));
  max_num_in_array<I>(matrix_a.rows + 1, row_ptr_c, device_max_row_num);
  KOKKOS_HIP_CHECK(hipMemcpy(max_row_num, device_max_row_num, sizeof(I), hipMemcpyDeviceToHost))
  KOKKOS_HIP_CHECK(hipFree(device_max_row_num))

  // reduce than fan
  scan_in_place<I>(matrix_a.rows + 1, row_ptr_c);

  // set nnz of matrix c
  KOKKOS_HIP_CHECK(hipMemcpy(nnz_c, row_ptr_c + matrix_a.rows, sizeof(I), hipMemcpyDeviceToHost))

  // free memory
  KOKKOS_HIP_CHECK(hipFree(map_use_count))
  KOKKOS_HIP_CHECK(hipFree(map_values))
  KOKKOS_HIP_CHECK(hipFree(map_ids))
  KOKKOS_HIP_CHECK(hipFree(row_trunk_ptr))
}

#endif // SPGEMM_KOKKOS_SPGEMM_SYMBOLIC_IMPL_HPP