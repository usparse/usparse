#ifndef SPGEMM_KOKKOS_SPGEMM_SYMBOLIC_KERNEL_HPP
#define SPGEMM_KOKKOS_SPGEMM_SYMBOLIC_KERNEL_HPP

#include "compress_matrix.h"
#include "hash_map.h"
#include "hash_map_insert.hpp"
#include <hip/hip_runtime.h>

template <typename I, typename BitType, size_t VEC_SIZE, size_t WF_SIZE>
__global__ void row_nnz_kernel(const I m, const I *map_row_ptr, const BitType *map_values, const int *use_count,
                               I *row_ptr_c) {
  // thread id in global
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  // thread num in global
  const int global_thread_num = gridDim.x * blockDim.x;
  // vector id in global
  const int global_vector_id = global_thread_id / VEC_SIZE;
  // vector num in global
  const int global_vector_num = global_thread_num / VEC_SIZE;
  // thread id in vector
  const int vector_thread_id = global_thread_id % VEC_SIZE;
  // thread num in vector
  const int vector_thread_num = VEC_SIZE;

  for (int row = global_vector_id; row < m; row += global_vector_num) {
    const I idx_begin = map_row_ptr[row];
    const I idx_end = idx_begin + use_count[row];
    I non_zero_count = 0;
    for (int idx = idx_begin + vector_thread_id; idx < idx_end; idx += vector_thread_num) {
      non_zero_count += __builtin_popcount(map_values[idx]);
    }
    // reduce
    for (int i = VEC_SIZE >> 1; i > 0; i >>= 1) {
      non_zero_count += __shfl_down(non_zero_count, i, VEC_SIZE);
    }
    if (vector_thread_id == 0) {
      row_ptr_c[row + 1] = non_zero_count;
    }
  }
}

template <typename I, size_t VEC_SIZE, size_t WF_SIZE>
__global__ void trunk_nums_kernel(const size_t m, const I *row_ptr_a, const I *col_idx_ptr_a, const int *use_count,
                                  I *row_trunks_ptr) {
  using idtype_t = typename CompressMatrix<I>::IdType;
  using bittype_t = typename CompressMatrix<I>::BitSetEleType;
  // thread id in global
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  // thread num in global
  const int global_thread_num = gridDim.x * blockDim.x;
  // vector id in global
  const int global_vector_id = global_thread_id / VEC_SIZE;
  // vector num in global
  const int global_vector_num = global_thread_num / VEC_SIZE;
  // thread id in vector
  const int vector_thread_id = global_thread_id % VEC_SIZE;
  // thread num in vector
  const int vector_thread_num = VEC_SIZE;

  // each vector process a row
  for (int row_a = global_vector_id; row_a < m; row_a += global_vector_num) {
    const I row_idx_begin_a = row_ptr_a[row_a];
    const I row_idx_end_a = row_ptr_a[row_a + 1];

    I trunk_count = 0;
    for (int idx_a = row_idx_begin_a + vector_thread_id; idx_a < row_idx_end_a; idx_a += vector_thread_num) {
      const I row_b = col_idx_ptr_a[idx_a];
      trunk_count += use_count[row_b];
    }
    // reduce
    for (int i = VEC_SIZE >> 1; i > 0; i >>= 1) {
      trunk_count += __shfl_down(trunk_count, i, VEC_SIZE);
    }
    if (vector_thread_id == 0) {
      row_trunks_ptr[row_a + 1] = trunk_count;
    }
  }
}

template <typename I, size_t BitsLen, size_t VEC_SIZE, size_t WF_SIZE>
__global__ void compress_matrix_kernel(const size_t m, const I *row_ptr, const I *col_idx_ptr,
                                       typename CompressMatrix<I>::IdType *ids,
                                       typename CompressMatrix<I>::BitSetEleType *bits_cols, int *use_count) {
  using idtype_t = typename CompressMatrix<I>::IdType;
  using bittype_t = typename CompressMatrix<I>::BitSetEleType;

  // thread id in global
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  // thread num in global
  const int global_thread_num = gridDim.x * blockDim.x;
  // each thread process a row
  for (int row = global_thread_id; row < m; row += global_thread_num) {
    const I row_idx_begin = row_ptr[row];
    const I row_idx_end = row_ptr[row + 1];

    idtype_t *thread_mem_ids = ids + row_idx_begin;
    bittype_t *thread_mem_bits = bits_cols + row_idx_begin;
    idtype_t count = -1;
    idtype_t id = 0xffffffff;
    for (int idx = row_idx_begin; idx < row_idx_end; idx++) {
      const I col = col_idx_ptr[idx];
      const idtype_t my_id = col / BitsLen;
      const bittype_t my_bit = 0x1 << (BitsLen - 1 - (col % BitsLen));
      if (my_id == id) {
        thread_mem_bits[count] |= my_bit;
      } else {
        count++;
        thread_mem_ids[count] = my_id;
        thread_mem_bits[count] = my_bit;
        id = my_id;
      }
    }
    use_count[row] = count + 1;
  }
}

template <typename T, typename I, size_t THREADS, size_t VEC_SIZE, size_t WF_SIZE>
__global__ void spgemm_symbolic_vector_per_row_kernel(
    const size_t m, const I *row_ptr_a, const I *col_idx_ptr_a, const I *row_ptr_b,
    const typename CompressMatrix<I>::IdType *ids_ptr_b, const typename CompressMatrix<I>::BitSetEleType *col_idx_ptr_b,
    const int *use_count, const I *row_ptr_c, int *begins, int *nexts, typename CompressMatrix<I>::IdType *ids,
    typename CompressMatrix<I>::BitSetEleType *values, int *map_use_count) {
  using idtype_t = typename CompressMatrix<I>::IdType;
  using bittype_t = typename CompressMatrix<I>::BitSetEleType;
  // thread id in global
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  // thread num in global
  const int global_thread_num = gridDim.x * blockDim.x;
  // vec id in global
  const int global_vec_id = global_thread_id / VEC_SIZE;
  // vec num in global
  const int global_vec_num = global_thread_num / VEC_SIZE;
  // block id in global
  const int global_block_id = blockIdx.x;
  // thread id in block
  const int block_thread_id = threadIdx.x;
  // thread num in block
  constexpr int block_thread_num = THREADS;
  // vec id in block
  const int block_vec_id = threadIdx.x / VEC_SIZE;
  // vec num in block
  constexpr int block_vec_num = THREADS / VEC_SIZE;
  // thread id in vec
  const int vec_thread_id = global_thread_id % VEC_SIZE;
  // thread num in vec
  constexpr int vec_thread_num = VEC_SIZE;

  // each vec process one row
  __shared__ int mem_use_size[block_vec_num];
  int *vec_mem_use_size = mem_use_size + block_vec_id;

  for (int row_a = global_vec_id; row_a < m; row_a += global_vec_num) {
    const I row_idx_begin = row_ptr_a[row_a];
    const I row_idx_end = row_ptr_a[row_a + 1];

    int *vec_mem_begins = begins + HashMap<bittype_t, idtype_t>::Capacity * row_a;
    int *vec_mem_nexts = nexts + row_ptr_c[row_a];
    idtype_t *vec_mem_ids = ids + row_ptr_c[row_a];
    bittype_t *vec_mem_values = values + row_ptr_c[row_a];

    if (vec_thread_id == 0) {
      *vec_mem_use_size = 0;
    }
    int idx_a = row_idx_begin;
    for (; idx_a < row_idx_end; ++idx_a) {
      const I col_a = col_idx_ptr_a[idx_a];
      const I row_b = col_a;
      const I idx_begin_b = row_ptr_b[row_b];
      const I idx_end_b = row_ptr_b[row_b + 1];
      const idtype_t *vec_mem_id_b = ids_ptr_b + idx_begin_b;
      const bittype_t *vec_mem_col_b = col_idx_ptr_b + idx_begin_b;
      const int b_count = use_count[row_b];
      for (int j = vec_thread_id; j < b_count; j += vec_thread_num) {
        Ihashmap_vector_insert_or<bittype_t, idtype_t, HashMap<bittype_t, idtype_t>::HashMask>(
            vec_mem_begins, vec_mem_nexts, vec_mem_ids, vec_mem_values, vec_mem_id_b[j], vec_mem_col_b[j],
            vec_mem_use_size);
      }
    }
    if (vec_thread_id == 0) {
      map_use_count[row_a] = *vec_mem_use_size;
    }
  }
}

#endif // SPGEMM_KOKKOS_SPGEMM_SYMBOLIC_KERNEL_HPP