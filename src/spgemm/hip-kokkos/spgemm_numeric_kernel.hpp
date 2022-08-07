#ifndef SPGEMM_KOKKOS_SPGEMM_NUMERIC_KERNEL_H
#define SPGEMM_KOKKOS_SPGEMM_NUMERIC_KERNEL_H

#include "hash_map.h"
#include "hash_map_insert.hpp"
#include <cstddef>
#include <hip/hip_runtime.h>

// please make sure GLOBAL_BLOCK_NUM = ROW_TO_PROCESS
template <typename T, typename I, size_t N, size_t THREADS, size_t VEC_SIZE, size_t WF_SIZE>
__global__ void __launch_bounds__(THREADS)
    spgemm_numeric_vector_per_row_sort_one_pass_kernel(const T alpha, const size_t m, const size_t n,
                                                       const I *row_ptr_a, const I *col_idx_ptr_a,
                                                       const T *values_ptr_a, const I *row_ptr_b,
                                                       const I *col_idx_ptr_b, const T *values_ptr_b,
                                                       const I *row_ptr_c, I *ids, T *values) {
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
  constexpr size_t LDS_MAP_CAP = VEC_SIZE * N;
  // hash mask
  constexpr size_t LDS_HASH_MASK = LDS_MAP_CAP - 1;

  constexpr int BLOCK_LDS_SIZE = LDS_MAP_CAP * block_vec_num;
  __shared__ I shared_ids[BLOCK_LDS_SIZE];
  __shared__ T shared_values[BLOCK_LDS_SIZE];

  I *vec_lds_ids = shared_ids + LDS_MAP_CAP * block_vec_id;
  T *vec_lds_values = shared_values + LDS_MAP_CAP * block_vec_id;

  // init lds hashmap
  for (int i = block_thread_id; i < BLOCK_LDS_SIZE; i += block_thread_num) {
    shared_ids[i] = HashMap<T, I>::NullValue;
    shared_values[i] = 0;
  }
  __syncthreads();

  int row_a = global_vec_id;

  if (row_a >= m) {
    return;
  }
  const I row_idx_begin = row_ptr_a[row_a];
  const I row_idx_end = row_ptr_a[row_a + 1];

  I *vec_mem_ids_end = ids + row_ptr_c[row_a + 1];
  T *vec_mem_values_end = values + row_ptr_c[row_a + 1];

  for (int idx_a = row_idx_begin; idx_a < row_idx_end; ++idx_a) {
    const I col_a = col_idx_ptr_a[idx_a];
    const I row_b = col_a;
    const I idx_begin_b = row_ptr_b[row_b];
    const I idx_end_b = row_ptr_b[row_b + 1];
    for (int idx_b = idx_begin_b + vec_thread_id; idx_b < row_idx_end; idx_b += vec_thread_num) {
      const T tmp_val = values_ptr_a[idx_a] * values_ptr_b[idx_b];
      const I col_b = col_idx_ptr_b[idx_b];
      // insert to lds hashmap
      open_address_hashmap_vector_insert_add<T, I, LDS_HASH_MASK>(vec_lds_ids, vec_lds_values, col_b, tmp_val);
    }
  }
  for (int i = vec_thread_id; i < LDS_MAP_CAP; i += vec_thread_num) {
    const I base_id = vec_lds_ids[i];
    const T base_values = vec_lds_values[i];
    if (base_id == HashMap<T, I>::NullValue) {
      continue;
    }
    int bigger_count = 0;
    for (int j = 0; j < LDS_MAP_CAP; ++j) {
      if (vec_lds_ids[j] > base_id) {
        bigger_count++;
      }
    }
    *(vec_mem_ids_end - bigger_count - 1) = base_id;
    *(vec_mem_values_end - bigger_count - 1) = base_values * alpha;
  }
}

// please make sure GLOBAL_BLOCK_NUM = ROW_TO_PROCESS
template <typename T, typename I, size_t N, size_t THREADS, size_t WF_SIZE>
__global__ void __launch_bounds__(THREADS)
    spgemm_numeric_block_per_row_sort_one_pass_kernel(const T alpha, const size_t m, const size_t n, const I *row_ptr_a,
                                                      const I *col_idx_ptr_a, const T *values_ptr_a, const I *row_ptr_b,
                                                      const I *col_idx_ptr_b, const T *values_ptr_b, const I *row_ptr_c,
                                                      I *ids, T *values) {
  // thread id in global
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  // thread num in global
  const int global_thread_num = gridDim.x * blockDim.x;
  // block id in global
  const int global_block_id = blockIdx.x;
  // block num in global
  const int global_block_num = gridDim.x;
  // thread id in block
  const int block_thread_id = threadIdx.x;
  // thread num in block
  constexpr int block_thread_num = THREADS;
  // wavefront id in block
  const int block_wf_id = block_thread_id / WF_SIZE;
  // wavefront num in block
  constexpr int block_wf_num = THREADS / WF_SIZE;
  // thread id in wavefront
  const int wf_thread_id = block_thread_id % WF_SIZE;
  // thread num in wavefront
  const int wf_thread_num = WF_SIZE;
  // hashmap in LDS
  constexpr size_t LDS_MAP_CAP = THREADS * N;
  // hash mask
  constexpr size_t LDS_HASH_MASK = LDS_MAP_CAP - 1;

  constexpr int BLOCK_LDS_SIZE = LDS_MAP_CAP;

  __shared__ I shared_ids[BLOCK_LDS_SIZE];
  __shared__ T shared_values[BLOCK_LDS_SIZE];

  I *block_lds_ids = shared_ids;
  T *block_lds_values = shared_values;

  // init lds hashmap
  for (int i = block_thread_id; i < BLOCK_LDS_SIZE; i += block_thread_num) {
    shared_ids[i] = HashMap<T, I>::NullValue;
    shared_values[i] = 0;
  }
  __syncthreads();

  // each block process one row
  int row_a = global_block_id;

  if (row_a >= m) {
    return;
  }

  const I row_idx_begin = row_ptr_a[row_a];
  const I row_idx_end = row_ptr_a[row_a + 1];
  I *block_mem_ids_end = ids + row_ptr_c[row_a + 1];
  T *block_mem_values_end = values + row_ptr_c[row_a + 1];

  for (int idx_a = row_idx_begin + block_wf_id; idx_a < row_idx_end; idx_a += block_wf_num) {
    const I col_a = col_idx_ptr_a[idx_a];
    const I row_b = col_a;
    const I idx_begin_b = row_ptr_b[row_b];
    const I idx_end_b = row_ptr_b[row_b + 1];
    for (int idx_b = idx_begin_b + wf_thread_id; idx_b < idx_end_b; idx_b += wf_thread_num) {
      const T tmp_val = values_ptr_a[idx_a] * values_ptr_b[idx_b];
      const I col_b = col_idx_ptr_b[idx_b];
      // insert to lds hashmap
      open_address_hashmap_vector_insert_add<T, I, LDS_HASH_MASK>(block_lds_ids, block_lds_values, col_b, tmp_val);
    }
  }
  __syncthreads();
  // sort and write to device memory
  for (int i = block_thread_id; i < LDS_MAP_CAP; i += block_thread_num) {
    const I base_id = block_lds_ids[i];
    const T base_values = block_lds_values[i];
    if (base_id == HashMap<T, I>::NullValue) {
      continue;
    }
    int bigger_count = 0;
    for (int j = 0; j < LDS_MAP_CAP; ++j) {
      if (block_lds_ids[j] > base_id) {
        bigger_count++;
      }
    }
    *(block_mem_ids_end - bigger_count - 1) = base_id;
    *(block_mem_values_end - bigger_count - 1) = base_values * alpha;
  }
}

template <typename T, typename I, size_t N, size_t LDS_MAP_CAP, size_t LDS_HASH_MASK, size_t THREADS, size_t WF_SIZE>
__global__ void __launch_bounds__(THREADS)
    spgemm_numeric_block_per_row_kernel(const T alpha, const size_t m, const size_t n, const I *row_ptr_a,
                                        const I *col_idx_ptr_a, const T *values_ptr_a, const I *row_ptr_b,
                                        const I *col_idx_ptr_b, const T *values_ptr_b, const I *row_ptr_c, int *begins,
                                        int *nexts, I *ids, T *values) {
  // thread id in global
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  // thread num in global
  const int global_thread_num = gridDim.x * blockDim.x;
  // block id in global
  const int global_block_id = blockIdx.x;
  // block num in global
  const int global_block_num = gridDim.x;
  // thread id in block
  const int block_thread_id = threadIdx.x;
  // thread num in block
  constexpr int block_thread_num = THREADS;
  // hashmap in LDS
  constexpr int LDS_BEGINS_SIZE = LDS_MAP_CAP;
  constexpr int BLOCK_PROCESS_ONCE = THREADS;

  constexpr int NEXTS_SIZE_PER_BLOCK = BLOCK_PROCESS_ONCE * N;
  constexpr int LDS_NEXTS_SIZE = NEXTS_SIZE_PER_BLOCK;

  __shared__ int shared_begins[LDS_BEGINS_SIZE];
  __shared__ int shared_nexts[LDS_NEXTS_SIZE];
  __shared__ I shared_ids[LDS_NEXTS_SIZE];
  __shared__ T shared_values[LDS_NEXTS_SIZE];

  int *block_lds_begins = shared_begins;
  int *block_lds_nexts = shared_nexts;
  I *block_lds_ids = shared_ids;
  T *block_lds_values = shared_values;
  // init lds hashmap
  for (int i = block_thread_id; i < LDS_BEGINS_SIZE; i += block_thread_num) {
    shared_begins[i] = HashMap<T, I>::NullValue;
  }
  for (int i = block_thread_id; i < LDS_NEXTS_SIZE; i += block_thread_num) {
    shared_nexts[i] = HashMap<T, I>::NullValue;
  }
  __syncthreads();
  // each vec process one row
  __shared__ int block_lds_use_size;
  __shared__ int block_mem_use_size;

  for (int row_a = global_block_id; row_a < m; row_a += global_block_num) {
    const I row_idx_begin = row_ptr_a[row_a];
    const I row_idx_end = row_ptr_a[row_a + 1];

    int *block_mem_begins = begins + HashMap<T, I>::Capacity * row_a;
    int *block_mem_nexts = nexts + row_ptr_c[row_a];
    I *block_mem_ids = ids + row_ptr_c[row_a];
    T *block_mem_values = values + row_ptr_c[row_a];

    if (block_thread_id == 0) {
      block_lds_use_size = 0;
      block_mem_use_size = 0;
    }
    __syncthreads();
    for (int idx_a = row_idx_begin; idx_a < row_idx_end; ++idx_a) {
      const I col_a = col_idx_ptr_a[idx_a];
      const I row_b = col_a;
      const I idx_begin_b = row_ptr_b[row_b];
      const I idx_end_b = row_ptr_b[row_b + 1];
      const int block_rounds = (idx_end_b - idx_begin_b) / BLOCK_PROCESS_ONCE +
                               ((idx_end_b - idx_begin_b) % BLOCK_PROCESS_ONCE == 0 ? 0 : 1);
      for (int round = 0; round < block_rounds; ++round) {
        if (block_lds_use_size + BLOCK_PROCESS_ONCE > NEXTS_SIZE_PER_BLOCK) {
          // if lds hashmap is full, write to device memory hashmap
          const int end_offset = block_lds_use_size;
          __syncthreads();
          for (int i = block_thread_id; i < end_offset; i += block_thread_num) {
            linear_chain_hashmap_vector_insert_add<T, I, HashMap<T, I>::HashMask>(
                block_mem_begins, block_mem_nexts, block_mem_ids, block_mem_values, block_lds_ids[i],
                block_lds_values[i] * alpha, &block_mem_use_size);
          }
          __syncthreads();
          // reset lds hashmap
          for (int i = block_thread_id; i < LDS_MAP_CAP; i += block_thread_num) {
            block_lds_begins[i] = HashMap<T, I>::NullValue;
          }
          for (int i = block_thread_id; i < end_offset; i += block_thread_num) {
            block_lds_nexts[i] = HashMap<T, I>::NullValue;
          }
          if (block_thread_id == 0) {
            block_lds_use_size = 0;
          }
          __syncthreads();
        }
        __syncthreads();
        const int idx_b = idx_begin_b + BLOCK_PROCESS_ONCE * round + block_thread_id;
        if (idx_b < idx_end_b) {
          const T tmp_val = values_ptr_a[idx_a] * values_ptr_b[idx_b];
          const I col_b = col_idx_ptr_b[idx_b];
          // insert to lds hashmap
          linear_chain_hashmap_vector_insert_add<T, I, LDS_HASH_MASK>(
              block_lds_begins, block_lds_nexts, block_lds_ids, block_lds_values, col_b, tmp_val, &block_lds_use_size);
        }
        __syncthreads();
      }
    }
    __syncthreads();
    const int end_offset = block_lds_use_size;
    __syncthreads();
    for (int i = block_thread_id; i < end_offset; i += block_thread_num) {
      linear_chain_hashmap_vector_insert_add<T, I, HashMap<T, I>::HashMask>(
          block_mem_begins, block_mem_nexts, block_mem_ids, block_mem_values, block_lds_ids[i],
          block_lds_values[i] * alpha, &block_mem_use_size);
    }
    __syncthreads();
    // reset lds hashmap
    for (int i = block_thread_id; i < LDS_MAP_CAP; i += block_thread_num) {
      block_lds_begins[i] = HashMap<T, I>::NullValue;
    }
    for (int i = block_thread_id; i < end_offset; i += block_thread_num) {
      block_lds_nexts[i] = HashMap<T, I>::NullValue;
    }
    if (block_thread_id == 0) {
      block_lds_use_size = 0;
    }
    __syncthreads();
  }
}

template <typename T, typename I, size_t N, size_t LDS_MAP_CAP, size_t LDS_HASH_MASK, size_t THREADS, size_t WF_SIZE>
__global__ void __launch_bounds__(THREADS)
    spgemm_numeric_block_per_row_sort_kernel(const T alpha, const size_t m, const size_t n, const I *row_ptr_a,
                                             const I *col_idx_ptr_a, const T *values_ptr_a, const I *row_ptr_b,
                                             const I *col_idx_ptr_b, const T *values_ptr_b, const I *row_ptr_c, I *ids,
                                             T *values) {
  // thread id in global
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  // thread num in global
  const int global_thread_num = gridDim.x * blockDim.x;
  // block id in global
  const int global_block_id = blockIdx.x;
  // block num in global
  const int global_block_num = gridDim.x;
  // thread id in block
  const int block_thread_id = threadIdx.x;
  // thread num in block
  constexpr int block_thread_num = THREADS;
  // hashmap in LDS
  constexpr int LDS_BEGINS_SIZE = LDS_MAP_CAP;
  constexpr int BLOCK_PROCESS_ONCE = THREADS;

  constexpr int NEXTS_SIZE_PER_BLOCK = BLOCK_PROCESS_ONCE * N;
  constexpr int LDS_NEXTS_SIZE = NEXTS_SIZE_PER_BLOCK;

  __shared__ int shared_begins[LDS_BEGINS_SIZE];
  __shared__ int shared_nexts[LDS_NEXTS_SIZE];
  __shared__ I shared_ids[LDS_NEXTS_SIZE];
  __shared__ T shared_values[LDS_NEXTS_SIZE];

  int *block_lds_begins = shared_begins;
  int *block_lds_nexts = shared_nexts;
  I *block_lds_ids = shared_ids;
  T *block_lds_values = shared_values;
  // init lds hashmap
  for (int i = block_thread_id; i < LDS_BEGINS_SIZE; i += block_thread_num) {
    shared_begins[i] = HashMap<T, I>::NullValue;
  }
  for (int i = block_thread_id; i < LDS_NEXTS_SIZE; i += block_thread_num) {
    shared_nexts[i] = HashMap<T, I>::NullValue;
  }
  __syncthreads();
  // each vec process one row
  __shared__ int block_lds_use_size;

  for (int row_a = global_block_id; row_a < m; row_a += global_block_num) {
    const I row_idx_begin = row_ptr_a[row_a];
    const I row_idx_end = row_ptr_a[row_a + 1];

    I *block_mem_ids = ids + row_ptr_c[row_a];
    T *block_mem_values = values + row_ptr_c[row_a];

    if (block_thread_id == 0) {
      block_lds_use_size = 0;
    }
    __syncthreads();
    int idx_a = row_idx_begin;
    for (; idx_a < row_idx_end; ++idx_a) {
      const I col_a = col_idx_ptr_a[idx_a];
      const I row_b = col_a;
      const I idx_begin_b = row_ptr_b[row_b];
      const I idx_end_b = row_ptr_b[row_b + 1];
      const int block_rounds = (idx_end_b - idx_begin_b) / BLOCK_PROCESS_ONCE +
                               ((idx_end_b - idx_begin_b) % BLOCK_PROCESS_ONCE == 0 ? 0 : 1);
      __syncthreads();
      for (int round = 0; round < block_rounds; ++round) {
        const int idx_b = idx_begin_b + BLOCK_PROCESS_ONCE * round + block_thread_id;
        if (idx_b < idx_end_b) {
          const T tmp_val = values_ptr_a[idx_a] * values_ptr_b[idx_b];
          const I col_b = col_idx_ptr_b[idx_b];
          // insert to lds hashmap
          linear_chain_hashmap_vector_insert_add<T, I, LDS_HASH_MASK>(
              block_lds_begins, block_lds_nexts, block_lds_ids, block_lds_values, col_b, tmp_val, &block_lds_use_size);
        }
      }
      __syncthreads();
    }
    __syncthreads();
    // sort and write to device memory
    const int end_offset = block_lds_use_size;

    for (int i = block_thread_id; i < end_offset; i += block_thread_num) {
      const I base_id = block_lds_ids[i];
      const T base_values = block_lds_values[i];
      int smaller_count = 0;
      for (int j = 0; j < end_offset; ++j) {
        if (block_lds_ids[j] < base_id) {
          smaller_count++;
        }
      }
      block_mem_ids[smaller_count] = base_id;
      block_mem_values[smaller_count] = base_values * alpha;
    }
    __syncthreads();
    // reset lds hashmap
    for (int i = block_thread_id; i < LDS_MAP_CAP; i += block_thread_num) {
      block_lds_begins[i] = HashMap<T, I>::NullValue;
    }
    for (int i = block_thread_id; i < end_offset; i += block_thread_num) {
      block_lds_nexts[i] = HashMap<T, I>::NullValue;
    }
    if (block_thread_id == 0) {
      block_lds_use_size = 0;
    }
    __syncthreads();
  }
}

template <typename T, typename I, size_t N, size_t LDS_MAP_CAP, size_t LDS_HASH_MASK, size_t THREADS, size_t VEC_SIZE,
          size_t WF_SIZE>
__global__ void __launch_bounds__(THREADS)
    spgemm_numeric_vector_per_row_sort_kernel(const T alpha, const size_t m, const size_t n, const I *row_ptr_a,
                                              const I *col_idx_ptr_a, const T *values_ptr_a, const I *row_ptr_b,
                                              const I *col_idx_ptr_b, const T *values_ptr_b, const I *row_ptr_c, I *ids,
                                              T *values) {
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
  constexpr int VEC_PROCESS_ONCE = VEC_SIZE;

  constexpr int NEXTS_SIZE_PER_VEC = VEC_PROCESS_ONCE * N;
  constexpr int LDS_NEXTS_SIZE = NEXTS_SIZE_PER_VEC * block_vec_num;
  __shared__ int shared_begins[LDS_BEGINS_SIZE];
  __shared__ int shared_nexts[LDS_NEXTS_SIZE];
  __shared__ I shared_ids[LDS_NEXTS_SIZE];
  __shared__ T shared_values[LDS_NEXTS_SIZE];

  int *vec_lds_begins = shared_begins + LDS_MAP_CAP * block_vec_id;
  int *vec_lds_nexts = shared_nexts + NEXTS_SIZE_PER_VEC * block_vec_id;
  I *vec_lds_ids = shared_ids + NEXTS_SIZE_PER_VEC * block_vec_id;
  T *vec_lds_values = shared_values + NEXTS_SIZE_PER_VEC * block_vec_id;

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
          (idx_end_b - idx_begin_b) / VEC_PROCESS_ONCE + ((idx_end_b - idx_begin_b) % VEC_PROCESS_ONCE == 0 ? 0 : 1);
      for (int round = 0; round < vec_rounds; ++round) {
        const int idx_b = idx_begin_b + VEC_PROCESS_ONCE * round + vec_thread_id;
        if (idx_b < idx_end_b) {
          const T tmp_val = values_ptr_a[idx_a] * values_ptr_b[idx_b];
          const I col_b = col_idx_ptr_b[idx_b];
          // insert to lds hashmap
          linear_chain_hashmap_vector_insert_add<T, I, LDS_HASH_MASK>(vec_lds_begins, vec_lds_nexts, vec_lds_ids,
                                                                      vec_lds_values, col_b, tmp_val, vec_lds_use_size);
        }
      }
    }
    // sort and write to device memory
    const int end_offset = *vec_lds_use_size;

    for (int i = vec_thread_id; i < end_offset; i += vec_thread_num) {
      const I base_id = vec_lds_ids[i];
      const T base_values = vec_lds_values[i];
      int smaller_count = 0;
      for (int j = 0; j < end_offset; ++j) {
        if (vec_lds_ids[j] < base_id) {
          smaller_count++;
        }
      }
      vec_mem_ids[smaller_count] = base_id;
      vec_mem_values[smaller_count] = base_values * alpha;
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

template <typename T, typename I, size_t N, size_t LDS_MAP_CAP, size_t LDS_HASH_MASK, size_t THREADS, size_t VEC_SIZE,
          size_t WF_SIZE>
__global__ void __launch_bounds__(THREADS)
    spgemm_numeric_vector_per_row_kernel(const T alpha, const size_t m, const size_t n, const I *row_ptr_a,
                                         const I *col_idx_ptr_a, const T *values_ptr_a, const I *row_ptr_b,
                                         const I *col_idx_ptr_b, const T *values_ptr_b, const I *row_ptr_c, int *begins,
                                         int *nexts, I *ids, T *values) {
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
  constexpr int VEC_PROCESS_ONCE = VEC_SIZE;

  constexpr int NEXTS_SIZE_PER_VEC = VEC_PROCESS_ONCE * N;
  constexpr int LDS_NEXTS_SIZE = NEXTS_SIZE_PER_VEC * block_vec_num;
  __shared__ int shared_begins[LDS_BEGINS_SIZE];
  __shared__ int shared_nexts[LDS_NEXTS_SIZE];
  __shared__ I shared_ids[LDS_NEXTS_SIZE];
  __shared__ T shared_values[LDS_NEXTS_SIZE];
  int *vec_lds_begins = shared_begins + LDS_MAP_CAP * block_vec_id;
  int *vec_lds_nexts = shared_nexts + NEXTS_SIZE_PER_VEC * block_vec_id;
  I *vec_lds_ids = shared_ids + NEXTS_SIZE_PER_VEC * block_vec_id;
  T *vec_lds_values = shared_values + NEXTS_SIZE_PER_VEC * block_vec_id;
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
          (idx_end_b - idx_begin_b) / VEC_PROCESS_ONCE + ((idx_end_b - idx_begin_b) % VEC_PROCESS_ONCE == 0 ? 0 : 1);
      for (int round = 0; round < vec_rounds; ++round) {
        if (*vec_lds_use_size + VEC_PROCESS_ONCE > NEXTS_SIZE_PER_VEC) {
          // if lds hashmap is full, write to device memory hashmap
          const int end_offset = *vec_lds_use_size;
          for (int i = vec_thread_id; i < end_offset; i += vec_thread_num) {
            linear_chain_hashmap_vector_insert_add<T, I, HashMap<T, I>::HashMask>(
                vec_mem_begins, vec_mem_nexts, vec_mem_ids, vec_mem_values, vec_lds_ids[i], vec_lds_values[i] * alpha,
                vec_mem_use_size);
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
        const int idx_b = idx_begin_b + VEC_PROCESS_ONCE * round + vec_thread_id;
        if (idx_b < idx_end_b) {
          const T tmp_val = values_ptr_a[idx_a] * values_ptr_b[idx_b];
          const I col_b = col_idx_ptr_b[idx_b];
          // insert to lds hashmap
          linear_chain_hashmap_vector_insert_add<T, I, LDS_HASH_MASK>(vec_lds_begins, vec_lds_nexts, vec_lds_ids,
                                                                      vec_lds_values, col_b, tmp_val, vec_lds_use_size);
        }
      }
    }
    // write to device memory hashmap
    const int end_offset = *vec_lds_use_size;
    for (int i = vec_thread_id; i < end_offset; i += vec_thread_num) {
      linear_chain_hashmap_vector_insert_add<T, I, HashMap<T, I>::HashMask>(
          vec_mem_begins, vec_mem_nexts, vec_mem_ids, vec_mem_values, vec_lds_ids[i], vec_lds_values[i] * alpha,
          vec_mem_use_size);
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