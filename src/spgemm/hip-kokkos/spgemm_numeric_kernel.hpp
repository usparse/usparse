#ifndef SPGEMM_KOKKOS_SPGEMM_NUMERIC_KERNEL_H
#define SPGEMM_KOKKOS_SPGEMM_NUMERIC_KERNEL_H

#include "hash_map.h"
#include "hash_map_insert.hpp"
#include <cstddef>
#include <hip/hip_runtime.h>

template <typename I, size_t CHUNK_SIZE>
__global__ void create_bp_thread_per_row_kernel(const size_t m, const I *row_ptr, const I *col_idx_ptr, I *bp,
                                                const size_t BP_IDX_PER_ROW) {
  // thread id in global
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  // thread num in global
  const int global_thread_num = gridDim.x * blockDim.x;
  // each thread process a row
  for (int row = global_thread_id; row < m; row += global_thread_num) {
    I *row_bp_ptr = &bp[BP_IDX_PER_ROW * row];
    I base_bp = CHUNK_SIZE;
    int bp_idx = 0;
    const I row_idx_begin = row_ptr[row];
    const I row_idx_end = row_ptr[row + 1];

    row_bp_ptr[bp_idx++] = row_idx_begin;
    for (int idx = row_idx_begin; idx < row_idx_end; idx++) {
      const I col = col_idx_ptr[idx];
      while (col >= base_bp) {
        row_bp_ptr[bp_idx++] = idx;
        base_bp += CHUNK_SIZE;
      }
    }
    row_bp_ptr[bp_idx++] = row_idx_end;
  }
}

template <typename I, size_t CHUNK_SIZE>
__global__ void create_bp_wf_per_row_kernel(const size_t m, const I *row_ptr, const I *col_idx_ptr, I *bp,
                                            const size_t BP_IDX_PER_ROW) {
  // thread id in global
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  // thread num in global
  const int global_thread_num = gridDim.x * blockDim.x;
  // wf id in global
  const int global_wf_id = global_thread_id / __WF_SIZE__;
  // wf num in global
  const int global_wf_num = global_thread_num / __WF_SIZE__;
  // thread id in wf
  const int wf_thread_id = global_thread_id % __WF_SIZE__;
  // thread num in wf
  const int wf_thread_num = __WF_SIZE__;
  // each vector process a row
  for (int row = global_wf_id; row < m; row += global_wf_num) {
    I *row_bp_ptr = &bp[BP_IDX_PER_ROW * row];
    I base_bp = CHUNK_SIZE;
    int bp_idx = 0;
    const I row_idx_begin = row_ptr[row];
    const I row_idx_end = row_ptr[row + 1];

    if (wf_thread_id == 0) {
      row_bp_ptr[bp_idx++] = row_idx_begin;
    }

    for (int idx = row_idx_begin + wf_thread_id; idx < row_idx_end; idx += wf_thread_num) {
      const I col = col_idx_ptr[idx];
      bool flag = col >= base_bp;
      if (__any(flag)) {
        do {
          I tmp_idx = flag ? idx : row_idx_end;
          for (int i = __WF_SIZE__ >> 1; i > 0; i >>= 1) {
            tmp_idx = min(__shfl_down(tmp_idx, i), tmp_idx);
          }
          if (wf_thread_id == 0) {
            row_bp_ptr[bp_idx++] = tmp_idx;
          }
          base_bp += CHUNK_SIZE;
          flag = col >= base_bp;
        } while (__any(flag));
      }
    }
    if (wf_thread_id == 0) {
      row_bp_ptr[bp_idx++] = row_idx_end;
    }
  }
}

template <typename I, size_t CHUNK_SIZE, size_t BUFFER_SIZE>
__global__ void create_bp_wf_per_row_kernel(const size_t m, const I *row_ptr, const I *col_idx_ptr, I *bp,
                                            const size_t BP_IDX_PER_ROW) {
  // thread id in global
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  // thread num in global
  const int global_thread_num = gridDim.x * blockDim.x;
  // wf id in global
  const int global_wf_id = global_thread_id / __WF_SIZE__;
  // wf num in global
  const int global_wf_num = global_thread_num / __WF_SIZE__;
  // thread id in wf
  const int wf_thread_id = global_thread_id % __WF_SIZE__;
  // thread num in wf
  const int wf_thread_num = __WF_SIZE__;

  I local_bp[BUFFER_SIZE];
  // each vector process a row
  for (int row = global_wf_id; row < m; row += global_wf_num) {
    I *row_bp_ptr = &bp[BP_IDX_PER_ROW * row];
    I base_bp = CHUNK_SIZE;
    int bp_idx = 1;
    const I row_idx_begin = row_ptr[row];
    const I row_idx_end = row_ptr[row + 1];

    if (wf_thread_id == 0) {
      row_bp_ptr[0] = row_idx_begin;
    }
    // init local bp
    for (int i = 0; i < BUFFER_SIZE; i++) {
      local_bp[i] = row_idx_end;
    }
    for (int idx = row_idx_begin + wf_thread_id; idx < row_idx_end; idx += wf_thread_num) {
      const I col = col_idx_ptr[idx];
      while (col >= base_bp) {
        local_bp[bp_idx++] = idx;
        base_bp += CHUNK_SIZE;
      }
    }
    int max_bp_idx = bp_idx;
    for (int i = __WF_SIZE__ >> 1; i > 0; i >>= 1) {
      max_bp_idx = max(__shfl_down(max_bp_idx, i), max_bp_idx);
    }
    max_bp_idx = __shfl(max_bp_idx, 0);

    for (int i = 1; i < max_bp_idx; i++) {
      I bp_val = local_bp[i];
      for (int j = __WF_SIZE__ >> 1; j > 0; j >>= 1) {
        bp_val = min(__shfl_down(bp_val, j), bp_val);
      }
      if (wf_thread_id == 0) {
        row_bp_ptr[i] = bp_val;
      }
    }
    if (wf_thread_id == 0) {
      row_bp_ptr[max_bp_idx] = row_idx_end;
    }
  }
}

template <typename T, typename I, size_t N, size_t THREADS, size_t VEC_SIZE>
__global__ void __launch_bounds__(THREADS) spgemm_numeric_open_address_hashmap_vector_per_row_sorted_one_pass_kernel(
    const T alpha, const size_t m, const size_t n, const I *row_ptr_a, const I *col_idx_ptr_a, const T *values_ptr_a,
    const I *row_ptr_b, const I *col_idx_ptr_b, const T *values_ptr_b, const I *group_row_ptr, const int group_row_num,
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

  if (global_vec_id >= group_row_num) {
    return;
  }
  const int row_a = group_row_ptr[global_vec_id];

  __shared__ I shared_ids[BLOCK_LDS_SIZE];
  __shared__ T shared_values[BLOCK_LDS_SIZE];

  I *vec_lds_ids = shared_ids + LDS_MAP_CAP * block_vec_id;
  T *vec_lds_values = shared_values + LDS_MAP_CAP * block_vec_id;

  // init lds hashmap
  for (int i = vec_thread_id; i < LDS_MAP_CAP; i += vec_thread_num) {
    vec_lds_ids[i] = HashMap<T, I>::NullValue;
    vec_lds_values[i] = 0;
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
    for (int idx_b = idx_begin_b + vec_thread_id; idx_b < idx_end_b; idx_b += vec_thread_num) {
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

template <typename T, typename I, size_t N, size_t THREADS>
__global__ void __launch_bounds__(THREADS) spgemm_numeric_open_address_hashmap_block_per_row_sorted_one_pass_kernel(
    const T alpha, const size_t m, const size_t n, const I *row_ptr_a, const I *col_idx_ptr_a, const T *values_ptr_a,
    const I *row_ptr_b, const I *col_idx_ptr_b, const T *values_ptr_b, const I *group_row_ptr, const int group_row_num,
    const I *row_ptr_c, I *ids, T *values) {
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
  const int block_wf_id = block_thread_id / __WF_SIZE__;
  // wavefront num in block
  constexpr int block_wf_num = THREADS / __WF_SIZE__;
  // thread id in wavefront
  const int wf_thread_id = block_thread_id % __WF_SIZE__;
  // thread num in wavefront
  const int wf_thread_num = __WF_SIZE__;
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
  if (global_block_id >= group_row_num) {
    return;
  }
  const int row_a = group_row_ptr[global_block_id];

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

template <typename T, typename I, size_t N, size_t THREADS>
__global__ void __launch_bounds__(THREADS) spgemm_numeric_linear_chain_hashmap_block_per_row_sorted_one_pass_kernel(
    const T alpha, const size_t m, const size_t n, const I *row_ptr_a, const I *col_idx_ptr_a, const T *values_ptr_a,
    const I *row_ptr_b, const I *col_idx_ptr_b, const T *values_ptr_b, const I *group_row_ptr, const int group_row_num,
    const I *row_ptr_c, I *ids, T *values) {
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
  constexpr size_t LDS_MAP_CAP = THREADS * N;
  // hash mask
  constexpr size_t LDS_HASH_MASK = LDS_MAP_CAP - 1;
  constexpr int LDS_BEGINS_SIZE = LDS_MAP_CAP;
  constexpr int BLOCK_PROCESS_ONCE = THREADS;

  constexpr int NEXTS_SIZE_PER_BLOCK = BLOCK_PROCESS_ONCE * N;
  constexpr int LDS_NEXTS_SIZE = NEXTS_SIZE_PER_BLOCK;
  // each block process one row
  if (global_block_id >= group_row_num) {
    return;
  }

  const int row_a = group_row_ptr[global_block_id];

  __shared__ int shared_begins[LDS_BEGINS_SIZE];
  __shared__ int shared_nexts[LDS_NEXTS_SIZE];
  __shared__ I shared_ids[LDS_NEXTS_SIZE];
  __shared__ T shared_values[LDS_NEXTS_SIZE];
  __shared__ int block_lds_use_size;

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
  if (block_thread_id == 0) {
    block_lds_use_size = 0;
  }
  __syncthreads();

  const I row_idx_begin = row_ptr_a[row_a];
  const I row_idx_end = row_ptr_a[row_a + 1];

  I *block_mem_ids = ids + row_ptr_c[row_a];
  T *block_mem_values = values + row_ptr_c[row_a];

  for (int idx_a = row_idx_begin; idx_a < row_idx_end; ++idx_a) {
    const I col_a = col_idx_ptr_a[idx_a];
    const I row_b = col_a;
    const I idx_begin_b = row_ptr_b[row_b];
    const I idx_end_b = row_ptr_b[row_b + 1];
    const int block_rounds =
        (idx_end_b - idx_begin_b) / BLOCK_PROCESS_ONCE + ((idx_end_b - idx_begin_b) % BLOCK_PROCESS_ONCE == 0 ? 0 : 1);
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
}

template <typename T, typename I, size_t N, size_t THREADS, size_t VEC_SIZE>
__global__ void __launch_bounds__(THREADS) spgemm_numeric_linear_chain_hashmap_vector_per_row_sorted_one_pass_kernel(
    const T alpha, const size_t m, const size_t n, const I *row_ptr_a, const I *col_idx_ptr_a, const T *values_ptr_a,
    const I *row_ptr_b, const I *col_idx_ptr_b, const T *values_ptr_b, const I *group_row_ptr, const int group_row_num,
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

  if (global_vec_id >= group_row_num) {
    return;
  }
  const int row_a = group_row_ptr[global_vec_id];

  // hashmap in LDS
  constexpr size_t LDS_MAP_CAP = VEC_SIZE * N;
  // hash mask
  constexpr size_t LDS_HASH_MASK = LDS_MAP_CAP - 1;
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
  for (int i = vec_thread_id; i < LDS_MAP_CAP; i += vec_thread_num) {
    vec_lds_begins[i] = HashMap<T, I>::NullValue;
  }
  for (int i = vec_thread_id; i < NEXTS_SIZE_PER_VEC; i += vec_thread_num) {
    vec_lds_nexts[i] = HashMap<T, I>::NullValue;
  }
  // each vec process one row
  __shared__ int lds_use_size[block_vec_num];
  int *vec_lds_use_size = lds_use_size + block_vec_id;

  const I row_idx_begin = row_ptr_a[row_a];
  const I row_idx_end = row_ptr_a[row_a + 1];

  I *vec_mem_ids = ids + row_ptr_c[row_a];
  T *vec_mem_values = values + row_ptr_c[row_a];

  if (vec_thread_id == 0) {
    *vec_lds_use_size = 0;
  }
  for (int idx_a = row_idx_begin; idx_a < row_idx_end; ++idx_a) {
    const I col_a = col_idx_ptr_a[idx_a];
    const I row_b = col_a;
    const I idx_begin_b = row_ptr_b[row_b];
    const I idx_end_b = row_ptr_b[row_b + 1];
    for (int idx_b = idx_begin_b + vec_thread_id; idx_b < idx_end_b; idx_b += vec_thread_num) {
      const T tmp_val = values_ptr_a[idx_a] * values_ptr_b[idx_b];
      const I col_b = col_idx_ptr_b[idx_b];
      // insert to lds hashmap
      linear_chain_hashmap_vector_insert_add<T, I, LDS_HASH_MASK>(vec_lds_begins, vec_lds_nexts, vec_lds_ids,
                                                                  vec_lds_values, col_b, tmp_val, vec_lds_use_size);
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
}

template <typename T, typename I, size_t CHUNK_SIZE, size_t THREADS, size_t VEC_SIZE>
__global__ void __launch_bounds__(THREADS)
    spgemm_numeric_block_per_row_sorted_bp_kernel(const T alpha, const size_t m, const size_t n, const I *row_ptr_a,
                                                  const I *col_idx_ptr_a, const T *values_ptr_a, const I *row_ptr_b,
                                                  const I *col_idx_ptr_b, const T *values_ptr_b, const I *row_ptr_c,
                                                  I *ids, T *values, I *bp, const size_t BPS_PER_ROW,
                                                  const size_t BP_IDX_PER_ROW) {
  // block id in global
  const int global_block_id = blockIdx.x;
  // block num in global
  const int global_block_num = gridDim.x;
  // thread id in block
  const int block_thread_id = threadIdx.x;
  // thread num in block
  constexpr int block_thread_num = THREADS;
  // vec id in block
  const int block_vec_id = threadIdx.x / VEC_SIZE;
  // vec num in block
  constexpr int block_vec_num = THREADS / VEC_SIZE;
  // thread id in vec
  const int vec_thread_id = block_thread_id % VEC_SIZE;
  // thread num in vec
  constexpr int vec_thread_num = VEC_SIZE;
  // chunks
  __shared__ bool table[CHUNK_SIZE];
  __shared__ T data[CHUNK_SIZE];

  // block processes a row
  const int row_a = global_block_id;

  if (row_a >= m) {
    return;
  }

  const I row_idx_begin = row_ptr_a[row_a];
  const I row_idx_end = row_ptr_a[row_a + 1];
  I row_begin_c = row_ptr_c[row_a];

  for (int r = 0; r < BPS_PER_ROW; r++) {
    const int bp_idx = r;
    const int chunk_begin = bp_idx * CHUNK_SIZE;
    // init
    for (int i = block_thread_id; i < CHUNK_SIZE; i += block_thread_num) {
      table[i] = 0;
      data[i] = static_cast<T>(0);
    }
    __syncthreads();

    for (I idx_a = row_idx_begin + block_vec_id; idx_a < row_idx_end; idx_a += block_vec_num) {
      const I col_a = col_idx_ptr_a[idx_a];
      const I row_b = col_a;
      const T val_A = values_ptr_a[idx_a];

      I idx_begin_b = bp[row_b * BP_IDX_PER_ROW + bp_idx];
      I idx_end_b = bp[row_b * BP_IDX_PER_ROW + bp_idx + 1];

      for (I idx_b = idx_begin_b + vec_thread_id; idx_b < idx_end_b; idx_b += vec_thread_num) {
        const I col_b = col_idx_ptr_b[idx_b];
        table[col_b - chunk_begin] = 1;
        atomicAdd(data + col_b - chunk_begin, val_A * values_ptr_b[idx_b]);
      }
    }
    __syncthreads();
    // re-use shared memory
    int *scan_offsets = reinterpret_cast<int *>(data);
    // compress
    for (int i = block_thread_id; i < CHUNK_SIZE; i += block_thread_num) {
      bool has_nnz = table[i];
      T value = data[i];

      unsigned long long mask = __ballot(has_nnz);
      int nnz = __popcll(mask);

      unsigned long long lanemask_le = UINT64_MAX >> (sizeof(unsigned long long) * CHAR_BIT - (__lane_id() + 1));

      int offset = __popcll(lanemask_le & mask);
      __syncthreads();

      scan_offsets[block_thread_id / __WF_SIZE__] = nnz;
      __syncthreads();

      for (unsigned int j = 1; j < block_thread_num / __WF_SIZE__; ++j) {
        if (block_thread_id >= j * __WF_SIZE__) {
          offset += scan_offsets[j - 1];
        }
      }
      I idx = row_begin_c + offset - 1;

      if (has_nnz) {
        ids[idx] = chunk_begin + i;
        values[idx] = alpha * value;
      }

      if (block_thread_id == block_thread_num - 1) {
        scan_offsets[block_thread_num / __WF_SIZE__ - 1] = offset;
      }
      __syncthreads();
      row_begin_c += scan_offsets[block_thread_num / __WF_SIZE__ - 1];
    }
    __syncthreads();
  }
}

#endif // SPGEMM_KOKKOS_SPGEMM_NUMERIC_KERNEL_H
