#ifndef SPGEMM_KOKKOS_SPGEMM_NUMERIC_KERNEL_H
#define SPGEMM_KOKKOS_SPGEMM_NUMERIC_KERNEL_H

#include "../common/scan/scan_in_device.hpp"
#include "hash_map.h"
#include "hash_map_insert.hpp"
#include <cstddef>
#include <hip/hip_runtime.h>

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

template <typename T, typename I, size_t N, size_t THREADS, size_t ITEMS_PER_BUCKET = 8>
__global__ void __launch_bounds__(THREADS)
    spgemm_numeric_block_per_row_flash_sorted_one_pass_kernel(const T alpha, const size_t m, const size_t n,
                                                              const I *row_ptr_a, const I *col_idx_ptr_a,
                                                              const T *values_ptr_a, const I *row_ptr_b,
                                                              const I *col_idx_ptr_b, const T *values_ptr_b,
                                                              const I *group_row_ptr, const int group_row_num,
                                                              const I *row_ptr_c, I *ids, T *values) {
  // block id in global
  const int global_block_id = blockIdx.x;
  // thread id in block
  const int block_thread_id = threadIdx.x;
  // wf id in block
  const int block_wf_id = block_thread_id / __WF_SIZE__;
  // wf num in block
  constexpr int block_wf_num = THREADS / __WF_SIZE__;
  // thread num in block
  constexpr int block_thread_num = THREADS;
  // thread id in wf
  const int wf_thread_id = block_thread_id % __WF_SIZE__;
  // thread num in wf
  constexpr int wf_thread_num = __WF_SIZE__;
  // hashmap in LDS
  constexpr size_t LDS_MAP_CAP = THREADS * N;
  constexpr int LDS_BEGINS_SIZE = LDS_MAP_CAP;
  constexpr int LDS_NEXTS_SIZE = LDS_MAP_CAP;

  constexpr int FLASH_BUCKET_NUM = LDS_MAP_CAP / ITEMS_PER_BUCKET + (LDS_MAP_CAP % ITEMS_PER_BUCKET == 0 ? 0 : 1);
  const int FLASH_SORT_WIDTH = n / FLASH_BUCKET_NUM + (n % FLASH_BUCKET_NUM == 0 ? 0 : 1);
  // each block process one row
  if (global_block_id >= group_row_num) {
    return;
  }

  const int row_a = group_row_ptr[global_block_id];

  __shared__ int block_lds_begins[LDS_BEGINS_SIZE];
  __shared__ int block_lds_nexts[LDS_NEXTS_SIZE];
  __shared__ I block_lds_ids[LDS_NEXTS_SIZE];
  __shared__ T block_lds_values[LDS_NEXTS_SIZE];

  constexpr int PADDING_BUCKET_N = FLASH_BUCKET_NUM / __WF_SIZE__ + (FLASH_BUCKET_NUM % __WF_SIZE__ == 0 ? 0 : 1);
  constexpr size_t PADDING_FLASH_BUCKET_NUM = PADDING_BUCKET_N * __WF_SIZE__;
  __shared__ int flash_bucket_use_size[PADDING_FLASH_BUCKET_NUM];
  __shared__ int block_lds_use_size;
// init lds hashmap
#pragma unroll N
  for (int i = 0; i < N; ++i) {
    const int idx = block_thread_num * i + block_thread_id;
    block_lds_begins[idx] = HashMap<T, I>::NullValue;
    block_lds_nexts[idx] = HashMap<T, I>::NullValue;
  }
  if (block_thread_id == 0) {
    block_lds_use_size = 0;
  }

  const I row_idx_begin = row_ptr_a[row_a];
  const I row_idx_end = row_ptr_a[row_a + 1];

  I *const block_mem_ids = ids + row_ptr_c[row_a];
  T *const block_mem_values = values + row_ptr_c[row_a];
  I *const last_mem_id = ids + row_ptr_c[row_a + 1] - 1;
  T *const last_mem_val = values + row_ptr_c[row_a + 1] - 1;

  __syncthreads();
  for (int idx_a = row_idx_begin; idx_a < row_idx_end; ++idx_a) {
    const I col_a = col_idx_ptr_a[idx_a];
    const T val_A = values_ptr_a[idx_a];

    const I row_b = col_a;
    const I idx_begin_b = row_ptr_b[row_b];
    const I idx_end_b = row_ptr_b[row_b + 1];
    const int BLOCK_ROUNDS = (idx_end_b - idx_begin_b) / THREADS + ((idx_end_b - idx_begin_b) % THREADS == 0 ? 0 : 1);
    __syncthreads();
    if (BLOCK_ROUNDS == 0) {
      continue;
    }
    for (int r = 0; r < BLOCK_ROUNDS - 1; ++r) {
      const int idx_b = idx_begin_b + THREADS * r + block_thread_id;
      const T tmp_val = val_A * values_ptr_b[idx_b];
      const I col_b = col_idx_ptr_b[idx_b];
      const size_t hash_val = col_b / FLASH_SORT_WIDTH;
      // insert to lds hashmap
      linear_chain_hashmap_vector_insert_add<T, I>(block_lds_begins, block_lds_nexts, block_lds_ids, block_lds_values,
                                                   hash_val, col_b, tmp_val, &block_lds_use_size);
    }
    {
      const int r = BLOCK_ROUNDS - 1;
      const int idx_b = idx_begin_b + THREADS * r + block_thread_id;
      if (idx_b < idx_end_b) {
        const T tmp_val = val_A * values_ptr_b[idx_b];
        const I col_b = col_idx_ptr_b[idx_b];
        const size_t hash_val = col_b / FLASH_SORT_WIDTH;
        // insert to lds hashmap
        linear_chain_hashmap_vector_insert_add<T, I>(block_lds_begins, block_lds_nexts, block_lds_ids, block_lds_values,
                                                     hash_val, col_b, tmp_val, &block_lds_use_size);
      }
    }
  }
  __syncthreads();
  for (int i = block_thread_id; i < FLASH_BUCKET_NUM; i += block_thread_num) {
    int count = 0;
    for (int l = block_lds_begins[i]; l != HashMap<T, I>::NullValue; l = block_lds_nexts[l]) {
      count++;
    }
    flash_bucket_use_size[i] = count;
  }
  __syncthreads();
  // scan
  if (block_wf_id == 0) {
    if constexpr (PADDING_BUCKET_N == 1) {
      const int idx = wf_thread_id;
      int scan_val = flash_bucket_use_size[idx];
      common::scan_in_vector_device<int, __WF_SIZE__>(&scan_val);
      flash_bucket_use_size[idx] = scan_val;
    } else {
      int scan_sum = 0;
#pragma unroll PADDING_BUCKET_N
      for (int i = 0; i < PADDING_BUCKET_N; ++i) {
        const int idx = wf_thread_id + i * __WF_SIZE__;
        int scan_val = flash_bucket_use_size[idx];
        common::scan_in_vector_device<int, __WF_SIZE__>(&scan_val);
        scan_val += scan_sum;
        flash_bucket_use_size[idx] = scan_val;
        scan_sum = __shfl(scan_val, __WF_SIZE__ - 1);
      }
    }
  }
  __syncthreads();
  // flash sort
  const int end_offset = block_lds_use_size;
  for (int i = block_thread_id; i < end_offset; i += block_thread_num) {
    const I base_id = block_lds_ids[i];
    const T base_values = block_lds_values[i];
    const size_t hash_val = base_id / FLASH_SORT_WIDTH;
    int smaller_count = hash_val > 0 ? flash_bucket_use_size[hash_val - 1] : 0;
    for (int l = block_lds_begins[hash_val]; l != HashMap<T, I>::NullValue; l = block_lds_nexts[l]) {
      if (block_lds_ids[l] < base_id) {
        smaller_count++;
      }
    }
    if (smaller_count == end_offset - 1) {
      *last_mem_id = base_id;
      *last_mem_val = base_values * alpha;
      continue;
    }
    block_mem_ids[smaller_count] = base_id;
    block_mem_values[smaller_count] = base_values * alpha;
  }
}

template <typename T, typename I, size_t N, size_t THREADS, size_t VEC_SIZE, size_t ITEMS_PER_BUCKET = 8>
__global__ void __launch_bounds__(THREADS)
    spgemm_numeric_vector_per_row_flash_sorted_one_pass_kernel(const T alpha, const size_t m, const size_t n,
                                                               const I *row_ptr_a, const I *col_idx_ptr_a,
                                                               const T *values_ptr_a, const I *row_ptr_b,
                                                               const I *col_idx_ptr_b, const T *values_ptr_b,
                                                               const I *group_row_ptr, const int group_row_num,
                                                               const I *row_ptr_c, I *ids, T *values) {
  // thread id in global
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  // thread num in global
  const int global_thread_num = gridDim.x * blockDim.x;
  // vec id in global
  const int global_vec_id = global_thread_id / VEC_SIZE;
  // vec num in global
  const int global_vec_num = global_thread_num / VEC_SIZE;
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
  constexpr int BEGINS_SIZE_PER_VEC = LDS_MAP_CAP;
  constexpr int LDS_BEGINS_SIZE = LDS_MAP_CAP * block_vec_num;
  constexpr int NEXTS_SIZE_PER_VEC = LDS_MAP_CAP;
  constexpr int LDS_NEXTS_SIZE = LDS_MAP_CAP * block_vec_num;
  constexpr int VEC_PROCESS_ONCE = VEC_SIZE;

  constexpr int FLASH_BUCKET_NUM = LDS_MAP_CAP / ITEMS_PER_BUCKET + (LDS_MAP_CAP % ITEMS_PER_BUCKET == 0 ? 0 : 1);
  const int FLASH_SORT_WIDTH = n / FLASH_BUCKET_NUM + (n % FLASH_BUCKET_NUM == 0 ? 0 : 1);

  __shared__ int shared_begins[LDS_BEGINS_SIZE];
  __shared__ int shared_nexts[LDS_NEXTS_SIZE];
  __shared__ I shared_ids[LDS_NEXTS_SIZE];
  __shared__ T shared_values[LDS_NEXTS_SIZE];
  __shared__ int lds_use_size[block_vec_num];

  int *const vec_lds_begins = shared_begins + BEGINS_SIZE_PER_VEC * block_vec_id;
  int *const vec_lds_nexts = shared_nexts + NEXTS_SIZE_PER_VEC * block_vec_id;
  I *const vec_lds_ids = shared_ids + NEXTS_SIZE_PER_VEC * block_vec_id;
  T *const vec_lds_values = shared_values + NEXTS_SIZE_PER_VEC * block_vec_id;
  int *const vec_lds_use_size = lds_use_size + block_vec_id;

  constexpr int PADDING_BUCKET_N = FLASH_BUCKET_NUM / VEC_SIZE + (FLASH_BUCKET_NUM % VEC_SIZE == 0 ? 0 : 1);
  constexpr size_t PADDING_FLASH_BUCKET_NUM = PADDING_BUCKET_N * VEC_SIZE;
  __shared__ int shared_flash_bucket_use_size[PADDING_FLASH_BUCKET_NUM * block_vec_num];
  int *const flash_bucket_use_size = shared_flash_bucket_use_size + PADDING_FLASH_BUCKET_NUM * block_vec_id;

// init lds hashmap
#pragma unroll N
  for (int i = 0; i < N; ++i) {
    const int idx = vec_thread_id + VEC_SIZE * i;
    vec_lds_begins[idx] = HashMap<T, I>::NullValue;
    vec_lds_nexts[idx] = HashMap<T, I>::NullValue;
  }
  if (vec_thread_id == 0) {
    *vec_lds_use_size = 0;
  }
  const I row_idx_begin = row_ptr_a[row_a];
  const I row_idx_end = row_ptr_a[row_a + 1];

  I *const vec_mem_ids = ids + row_ptr_c[row_a];
  T *const vec_mem_values = values + row_ptr_c[row_a];

  for (int idx_a = row_idx_begin; idx_a < row_idx_end; ++idx_a) {
    const I col_a = col_idx_ptr_a[idx_a];
    const T val_A = values_ptr_a[idx_a];

    const I row_b = col_a;
    const I idx_begin_b = row_ptr_b[row_b];
    const I idx_end_b = row_ptr_b[row_b + 1];
    for (int idx_b = idx_begin_b + vec_thread_id; idx_b < idx_end_b; idx_b += vec_thread_num) {
      const T tmp_val = val_A * values_ptr_b[idx_b];
      const I col_b = col_idx_ptr_b[idx_b];
      const size_t hash_val = col_b / FLASH_SORT_WIDTH;
      // insert to lds hashmap
      linear_chain_hashmap_vector_insert_add<T, I>(vec_lds_begins, vec_lds_nexts, vec_lds_ids, vec_lds_values, hash_val,
                                                   col_b, tmp_val, vec_lds_use_size);
    }
  }
  // flash sort
  for (int i = vec_thread_id; i < FLASH_BUCKET_NUM; i += vec_thread_num) {
    int l = vec_lds_begins[i];
    int count = 0;
    for (; l != HashMap<T, I>::NullValue; l = vec_lds_nexts[l]) {
      count++;
    }
    flash_bucket_use_size[i] = count;
  }
  // scan
  if constexpr (PADDING_BUCKET_N == 1) {
    const int idx = vec_thread_id;
    int scan_val = flash_bucket_use_size[idx];
    common::scan_in_vector_device<int, VEC_SIZE>(&scan_val);
    flash_bucket_use_size[idx] = scan_val;
  } else {
    int scan_sum = 0;
#pragma unroll PADDING_BUCKET_N
    for (int i = 0; i < PADDING_BUCKET_N; ++i) {
      const int idx = vec_thread_id + i * VEC_SIZE;
      int scan_val = flash_bucket_use_size[idx];
      common::scan_in_vector_device<int, VEC_SIZE>(&scan_val);
      scan_val += scan_sum;
      flash_bucket_use_size[idx] = scan_val;
      scan_sum = __shfl(scan_val, VEC_SIZE - 1, VEC_SIZE);
    }
  }

  const int end_offset = *vec_lds_use_size;
  for (int i = vec_thread_id; i < end_offset; i += vec_thread_num) {
    const I base_id = vec_lds_ids[i];
    const T base_values = vec_lds_values[i];
    const size_t hash_val = base_id / FLASH_SORT_WIDTH;
    int smaller_count = hash_val > 0 ? flash_bucket_use_size[hash_val - 1] : 0;
    for (int l = vec_lds_begins[hash_val]; l != HashMap<T, I>::NullValue; l = vec_lds_nexts[l]) {
      if (vec_lds_ids[l] < base_id) {
        smaller_count++;
      }
    }
    vec_mem_ids[smaller_count] = base_id;
    vec_mem_values[smaller_count] = base_values * alpha;
  }
}

template <typename T, typename I, size_t N, size_t THREADS>
__global__ void __launch_bounds__(THREADS) spgemm_numeric_linear_chain_hashmap_block_per_row_sorted_one_pass_kernel(
    const T alpha, const size_t m, const size_t n, const I *row_ptr_a, const I *col_idx_ptr_a, const T *values_ptr_a,
    const I *row_ptr_b, const I *col_idx_ptr_b, const T *values_ptr_b, const I *group_row_ptr, const int group_row_num,
    const I *row_ptr_c, I *ids, T *values) {
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
  constexpr size_t LDS_HASH_MASK = LDS_MAP_CAP - 1;
  constexpr int LDS_BEGINS_SIZE = LDS_MAP_CAP;
  constexpr int LDS_NEXTS_SIZE = LDS_MAP_CAP;
  // each block process one row
  if (global_block_id >= group_row_num) {
    return;
  }
  const int row_a = group_row_ptr[global_block_id];

  __shared__ int block_lds_begins[LDS_BEGINS_SIZE];
  __shared__ int block_lds_nexts[LDS_NEXTS_SIZE];
  __shared__ I block_lds_ids[LDS_NEXTS_SIZE];
  __shared__ T block_lds_values[LDS_NEXTS_SIZE];
  __shared__ int block_lds_use_size;
// init lds hashmap
#pragma unroll N
  for (int i = 0; i < N; ++i) {
    const int idx = block_thread_id + block_thread_num * i;
    block_lds_begins[idx] = HashMap<T, I>::NullValue;
    block_lds_nexts[idx] = HashMap<T, I>::NullValue;
  }
  if (block_thread_id == 0) {
    block_lds_use_size = 0;
  }
  const I row_idx_begin = row_ptr_a[row_a];
  const I row_idx_end = row_ptr_a[row_a + 1];

  I *block_mem_ids = ids + row_ptr_c[row_a];
  T *block_mem_values = values + row_ptr_c[row_a];

  for (int idx_a = row_idx_begin; idx_a < row_idx_end; ++idx_a) {
    const I col_a = col_idx_ptr_a[idx_a];
    const T val_A = values_ptr_a[idx_a];

    const I row_b = col_a;
    const I idx_begin_b = row_ptr_b[row_b];
    const I idx_end_b = row_ptr_b[row_b + 1];
    const int BLOCK_ROUNDS = (idx_end_b - idx_begin_b) / THREADS + ((idx_end_b - idx_begin_b) % THREADS == 0 ? 0 : 1);
    __syncthreads();
    if (BLOCK_ROUNDS == 0) {
      continue;
    }
    for (int r = 0; r < BLOCK_ROUNDS - 1; ++r) {
      const int idx_b = idx_begin_b + THREADS * r + block_thread_id;
      const T tmp_val = val_A * values_ptr_b[idx_b];
      const I col_b = col_idx_ptr_b[idx_b];
      // insert to lds hashmap
      linear_chain_hashmap_vector_insert_add<T, I, LDS_HASH_MASK>(
          block_lds_begins, block_lds_nexts, block_lds_ids, block_lds_values, col_b, tmp_val, &block_lds_use_size);
    }
    {
      const int r = BLOCK_ROUNDS - 1;
      const int idx_b = idx_begin_b + THREADS * r + block_thread_id;
      if (idx_b < idx_end_b) {
        const T tmp_val = val_A * values_ptr_b[idx_b];
        const I col_b = col_idx_ptr_b[idx_b];
        // insert to lds hashmap
        linear_chain_hashmap_vector_insert_add<T, I, LDS_HASH_MASK>(
            block_lds_begins, block_lds_nexts, block_lds_ids, block_lds_values, col_b, tmp_val, &block_lds_use_size);
      }
    }
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
  constexpr size_t LDS_HASH_MASK = LDS_MAP_CAP - 1;

  constexpr int BEGINS_SIZE_PER_VEC = LDS_MAP_CAP;
  constexpr int LDS_BEGINS_SIZE = BEGINS_SIZE_PER_VEC * block_vec_num;
  constexpr int NEXTS_SIZE_PER_VEC = LDS_MAP_CAP;
  constexpr int LDS_NEXTS_SIZE = NEXTS_SIZE_PER_VEC * block_vec_num;

  constexpr int VEC_PROCESS_ONCE = VEC_SIZE;

  __shared__ int shared_begins[LDS_BEGINS_SIZE];
  __shared__ int shared_nexts[LDS_NEXTS_SIZE];
  __shared__ I shared_ids[LDS_NEXTS_SIZE];
  __shared__ T shared_values[LDS_NEXTS_SIZE];
  __shared__ int lds_use_size[block_vec_num];

  int *vec_lds_begins = shared_begins + BEGINS_SIZE_PER_VEC * block_vec_id;
  int *vec_lds_nexts = shared_nexts + NEXTS_SIZE_PER_VEC * block_vec_id;
  I *vec_lds_ids = shared_ids + NEXTS_SIZE_PER_VEC * block_vec_id;
  T *vec_lds_values = shared_values + NEXTS_SIZE_PER_VEC * block_vec_id;
  int *vec_lds_use_size = lds_use_size + block_vec_id;

// init lds hashmap
#pragma unroll N
  for (int i = 0; i < N; ++i) {
    const int idx = vec_thread_id + vec_thread_num * i;
    vec_lds_begins[idx] = HashMap<T, I>::NullValue;
    vec_lds_nexts[idx] = HashMap<T, I>::NullValue;
  }
  if (vec_thread_id == 0) {
    *vec_lds_use_size = 0;
  }
  // each vec process one row
  const I row_idx_begin = row_ptr_a[row_a];
  const I row_idx_end = row_ptr_a[row_a + 1];
  I *vec_mem_ids = ids + row_ptr_c[row_a];
  T *vec_mem_values = values + row_ptr_c[row_a];
  I *last_mem_id = ids + row_ptr_c[row_a + 1] - 1;
  T *last_mem_val = values + row_ptr_c[row_a + 1] - 1;

  for (int idx_a = row_idx_begin; idx_a < row_idx_end; ++idx_a) {
    const I col_a = col_idx_ptr_a[idx_a];
    const T val_A = values_ptr_a[idx_a];

    const I row_b = col_a;
    const I idx_begin_b = row_ptr_b[row_b];
    const I idx_end_b = row_ptr_b[row_b + 1];
    for (int idx_b = idx_begin_b + vec_thread_id; idx_b < idx_end_b; idx_b += vec_thread_num) {
      const T tmp_val = val_A * values_ptr_b[idx_b];
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
    if (smaller_count == end_offset - 1) {
      *last_mem_id = base_id;
      *last_mem_val = base_values * alpha;
      continue;
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
  if (global_block_id >= m) {
    return;
  }
  const int row_a = global_block_id;
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
