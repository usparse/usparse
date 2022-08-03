#ifndef SPGEMM_KOKKOS_SPGEMM_NUMERIC_KERNEL_H
#define SPGEMM_KOKKOS_SPGEMM_NUMERIC_KERNEL_H

#include "hash_map.h"
#include "hash_map_insert.hpp"
#include <hip/hip_runtime.h>

template <typename T, typename I, size_t N, size_t LDS_MAP_CAP, size_t LDS_HASH_MASK, size_t THREADS, size_t VEC_SIZE,
          size_t WF_SIZE>
__global__ void spgemm_numeric_vector_per_row_kernel(const T alpha, const size_t m, const size_t n, const I *row_ptr_a,
                                                     const I *col_idx_ptr_a, const T *values_ptr_a, const I *row_ptr_b,
                                                     const I *col_idx_ptr_b, const T *values_ptr_b, const I *row_ptr_c,
                                                     int *begins, int *nexts, I *ids, T *values) {
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
  // hashmap in LDS
  constexpr int LDS_BEGINS_SIZE = LDS_MAP_CAP * block_vec_num;
  constexpr int WF_PROCESS_ONCE = VEC_SIZE;

  constexpr int NEXTS_SIZE_PER_WF = WF_PROCESS_ONCE * N;
  constexpr int LDS_NEXTS_SIZE = NEXTS_SIZE_PER_WF * block_vec_num;
  __shared__ int shared_begins[LDS_BEGINS_SIZE];
  __shared__ int shared_nexts[LDS_NEXTS_SIZE];
  __shared__ I shared_ids[LDS_NEXTS_SIZE];
  __shared__ T shared_values[LDS_NEXTS_SIZE];
  int *vec_lds_begins = shared_begins + LDS_MAP_CAP * block_vec_id;
  int *vec_lds_nexts = shared_nexts + NEXTS_SIZE_PER_WF * block_vec_id;
  I *vec_lds_ids = shared_ids + NEXTS_SIZE_PER_WF * block_vec_id;
  T *vec_lds_values = shared_values + NEXTS_SIZE_PER_WF * block_vec_id;
  // init lds hashmap
  for (int i = block_thread_id; i < LDS_BEGINS_SIZE; i += block_thread_num) {
    shared_begins[i] = HashMap<T, I>::NullValue;
  }
  for (int i = block_thread_id; i < LDS_NEXTS_SIZE; i += block_thread_num) {
    shared_nexts[i] = HashMap<T, I>::NullValue;
  }
  __syncthreads();
  // each vec process one row
  __shared__ int lds_use_size[block_vec_num];
  __shared__ int mem_use_size[block_vec_num];
  int *vec_lds_use_size = lds_use_size + block_vec_id;
  int *vec_mem_use_size = mem_use_size + block_vec_id;

  for (int row_a = global_vec_id; row_a < m; row_a += global_vec_num) {
    const I row_idx_begin = row_ptr_a[row_a];
    const I row_idx_end = row_ptr_a[row_a + 1];

    int *vec_mem_begins = begins + HashMap<T, I>::Capacity * row_a;
    int *vec_mem_nexts = nexts + row_ptr_c[row_a];
    I *vec_mem_ids = ids + row_ptr_c[row_a];
    T *vec_mem_values = values + row_ptr_c[row_a];

    if (vec_thread_id == 0) {
      *vec_lds_use_size = 0;
      *vec_mem_use_size = 0;
    }
    int idx_a = row_idx_begin;
    for (; idx_a < row_idx_end; ++idx_a) {
      const I col_a = col_idx_ptr_a[idx_a];
      const I row_b = col_a;
      const I idx_begin_b = row_ptr_b[row_b];
      const I idx_end_b = row_ptr_b[row_b + 1];
      const int vec_rounds =
          (idx_end_b - idx_begin_b) / WF_PROCESS_ONCE + ((idx_end_b - idx_begin_b) % WF_PROCESS_ONCE == 0 ? 0 : 1);
      for (int round = 0; round < vec_rounds; ++round) {
        if (*vec_lds_use_size + WF_PROCESS_ONCE > NEXTS_SIZE_PER_WF) {
          // if lds hashmap is full, write to device memory hashmap
          const int end_offset = *vec_lds_use_size;
          for (int i = vec_thread_id; i < end_offset; i += vec_thread_num) {
            Ihashmap_vector_insert_add<T, I, HashMap<T, I>::HashMask>(vec_mem_begins, vec_mem_nexts, vec_mem_ids,
                                                                      vec_mem_values, vec_lds_ids[i],
                                                                      vec_lds_values[i] * alpha, vec_mem_use_size);
          }
          // reset lds hashmap
          for (int i = vec_thread_id; i < LDS_MAP_CAP; i += vec_thread_num) {
            vec_lds_begins[i] = HashMap<T, I>::NullValue;
          }
          for (int i = vec_thread_id; i < end_offset; i += vec_thread_num) {
            vec_lds_nexts[i] = HashMap<T, I>::NullValue;
          }
          if (vec_thread_id == 0) {
            *vec_lds_use_size = 0;
          }
        }
        const int idx_b = idx_begin_b + WF_PROCESS_ONCE * round + vec_thread_id;
        if (idx_b < idx_end_b) {
          const T tmp_val = values_ptr_a[idx_a] * values_ptr_b[idx_b];
          const I col_b = col_idx_ptr_b[idx_b];
          // insert to lds hashmap
          Ihashmap_vector_insert_add<T, I, LDS_HASH_MASK>(vec_lds_begins, vec_lds_nexts, vec_lds_ids, vec_lds_values,
                                                          col_b, tmp_val, vec_lds_use_size);
        }
      }
    }
    // write to device memory hashmap
    const int end_offset = *vec_lds_use_size;
    for (int i = vec_thread_id; i < end_offset; i += vec_thread_num) {
      Ihashmap_vector_insert_add<T, I, HashMap<T, I>::HashMask>(vec_mem_begins, vec_mem_nexts, vec_mem_ids,
                                                                vec_mem_values, vec_lds_ids[i],
                                                                vec_lds_values[i] * alpha, vec_mem_use_size);
    }
    // reset lds hashmap
    for (int i = vec_thread_id; i < LDS_MAP_CAP; i += vec_thread_num) {
      vec_lds_begins[i] = HashMap<T, I>::NullValue;
    }
    for (int i = vec_thread_id; i < end_offset; i += vec_thread_num) {
      vec_lds_nexts[i] = HashMap<T, I>::NullValue;
    }
    if (vec_thread_id == 0) {
      *vec_lds_use_size = 0;
    }
  }
}

#endif // SPGEMM_KOKKOS_SPGEMM_NUMERIC_KERNEL_H