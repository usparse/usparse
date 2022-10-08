#ifndef SPGEMM_KOKKOS_SPGEMM_NUMERIC_KERNEL_H
#define SPGEMM_KOKKOS_SPGEMM_NUMERIC_KERNEL_H

#include "../common/scan/scan_in_device.hpp"
#include "hash_map_insert.hpp"
#include <cstddef>
#include <hip/hip_runtime.h>

struct FlashSortSingleScanTag {};
struct FlashSortMultipleScanTag {};

template <typename T, typename I, size_t CHUNK_SIZE, size_t THREADS, size_t VEC_SIZE>
__global__ void __launch_bounds__(THREADS)
    spgemm_numeric_block_per_row_sorted_bp_kernel(const T alpha, const size_t m, const size_t n, const I *row_ptr_a,
                                                  const I *col_idx_ptr_a, const T *values_ptr_a, const I *row_ptr_b,
                                                  const I *col_idx_ptr_b, const T *values_ptr_b, const I *row_ptr_c,
                                                  I *ids, T *values, const I *bp, const size_t BPS_PER_ROW,
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

template <typename T, typename I, size_t N, size_t THREADS, size_t VEC_SIZE, size_t ITEMS_PER_BUCKET>
__global__ void __launch_bounds__(THREADS) spgemm_numeric_vector_per_row_flash_sorted_one_pass_kernel(
    const T alpha, const size_t m, const size_t n, const I *row_ptr_a, const I *col_idx_ptr_a, const T *values_ptr_a,
    const I *row_ptr_b, const I *col_idx_ptr_b, const T *values_ptr_b, const I *group_row_ptr, const int group_row_num,
    const I *row_ptr_c, I *ids, T *values, FlashSortSingleScanTag) {
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
  constexpr int HASH_MAP_NULL_VALUE = 0xffffffff;

  constexpr int FLASH_BUCKET_NUM = LDS_MAP_CAP / ITEMS_PER_BUCKET + (LDS_MAP_CAP % ITEMS_PER_BUCKET == 0 ? 0 : 1);
  const int FLASH_SORT_WIDTH = n / FLASH_BUCKET_NUM + (n % FLASH_BUCKET_NUM == 0 ? 0 : 1);

  __shared__ int shared_begins[FLASH_BUCKET_NUM];
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
    vec_lds_nexts[idx] = HASH_MAP_NULL_VALUE;
  }
  for (int i = vec_thread_id; i < FLASH_BUCKET_NUM; i += VEC_SIZE) {
    vec_lds_begins[i] = HASH_MAP_NULL_VALUE;
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
      linear_chain_hashmap_vector_insert_add<T, I, HASH_MAP_NULL_VALUE>(
          vec_lds_begins, vec_lds_nexts, vec_lds_ids, vec_lds_values, hash_val, col_b, tmp_val, vec_lds_use_size);
    }
  }
  // flash sort
  for (int i = vec_thread_id; i < FLASH_BUCKET_NUM; i += vec_thread_num) {
    int l = vec_lds_begins[i];
    int count = 0;
    for (; l != HASH_MAP_NULL_VALUE; l = vec_lds_nexts[l]) {
      count++;
    }
    flash_bucket_use_size[i] = count;
  }
  // scan
  const int idx = vec_thread_id;
  int scan_val = flash_bucket_use_size[idx];
  common::scan_in_vector_device<int, VEC_SIZE>(&scan_val);
  flash_bucket_use_size[idx] = scan_val;

  const int end_offset = *vec_lds_use_size;
  for (int i = vec_thread_id; i < end_offset; i += vec_thread_num) {
    const I base_id = vec_lds_ids[i];
    const T base_values = vec_lds_values[i];
    const size_t hash_val = base_id / FLASH_SORT_WIDTH;
    int smaller_count = hash_val > 0 ? flash_bucket_use_size[hash_val - 1] : 0;
    for (int l = vec_lds_begins[hash_val]; l != HASH_MAP_NULL_VALUE; l = vec_lds_nexts[l]) {
      if (vec_lds_ids[l] < base_id) {
        smaller_count++;
      }
    }
    vec_mem_ids[smaller_count] = base_id;
    vec_mem_values[smaller_count] = base_values * alpha;
  }
}

template <typename T, typename I, size_t N, size_t THREADS, size_t VEC_SIZE, size_t ITEMS_PER_BUCKET>
__global__ void __launch_bounds__(THREADS) spgemm_numeric_vector_per_row_flash_sorted_one_pass_kernel(
    const T alpha, const size_t m, const size_t n, const I *row_ptr_a, const I *col_idx_ptr_a, const T *values_ptr_a,
    const I *row_ptr_b, const I *col_idx_ptr_b, const T *values_ptr_b, const I *group_row_ptr, const int group_row_num,
    const I *row_ptr_c, I *ids, T *values, FlashSortMultipleScanTag) {
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
  constexpr int HASH_MAP_NULL_VALUE = 0xffffffff;

  constexpr int FLASH_BUCKET_NUM = LDS_MAP_CAP / ITEMS_PER_BUCKET + (LDS_MAP_CAP % ITEMS_PER_BUCKET == 0 ? 0 : 1);
  const int FLASH_SORT_WIDTH = n / FLASH_BUCKET_NUM + (n % FLASH_BUCKET_NUM == 0 ? 0 : 1);

  __shared__ int shared_begins[FLASH_BUCKET_NUM];
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
    vec_lds_nexts[idx] = HASH_MAP_NULL_VALUE;
  }
  for (int i = vec_thread_id; i < FLASH_BUCKET_NUM; i += VEC_SIZE) {
    vec_lds_begins[i] = HASH_MAP_NULL_VALUE;
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
      linear_chain_hashmap_vector_insert_add<T, I, HASH_MAP_NULL_VALUE>(
          vec_lds_begins, vec_lds_nexts, vec_lds_ids, vec_lds_values, hash_val, col_b, tmp_val, vec_lds_use_size);
    }
  }
  // flash sort
  for (int i = vec_thread_id; i < FLASH_BUCKET_NUM; i += vec_thread_num) {
    int l = vec_lds_begins[i];
    int count = 0;
    for (; l != HASH_MAP_NULL_VALUE; l = vec_lds_nexts[l]) {
      count++;
    }
    flash_bucket_use_size[i] = count;
  }
  // scan
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

  const int end_offset = *vec_lds_use_size;
  for (int i = vec_thread_id; i < end_offset; i += vec_thread_num) {
    const I base_id = vec_lds_ids[i];
    const T base_values = vec_lds_values[i];
    const size_t hash_val = base_id / FLASH_SORT_WIDTH;
    int smaller_count = hash_val > 0 ? flash_bucket_use_size[hash_val - 1] : 0;
    for (int l = vec_lds_begins[hash_val]; l != HASH_MAP_NULL_VALUE; l = vec_lds_nexts[l]) {
      if (vec_lds_ids[l] < base_id) {
        smaller_count++;
      }
    }
    vec_mem_ids[smaller_count] = base_id;
    vec_mem_values[smaller_count] = base_values * alpha;
  }
}

template <typename T, typename I, size_t N, size_t THREADS, size_t ITEMS_PER_BUCKET>
__global__ void __launch_bounds__(THREADS) spgemm_numeric_block_per_row_flash_sorted_one_pass_kernel(
    const T alpha, const size_t m, const size_t n, const I *row_ptr_a, const I *col_idx_ptr_a, const T *values_ptr_a,
    const I *row_ptr_b, const I *col_idx_ptr_b, const T *values_ptr_b, const I *group_row_ptr, const int group_row_num,
    const I *row_ptr_c, I *ids, T *values, FlashSortSingleScanTag) {
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
  constexpr int HASH_MAP_NULL_VALUE = 0xffffffff;

  constexpr int FLASH_BUCKET_NUM = LDS_MAP_CAP / ITEMS_PER_BUCKET + (LDS_MAP_CAP % ITEMS_PER_BUCKET == 0 ? 0 : 1);
  const int FLASH_SORT_WIDTH = n / FLASH_BUCKET_NUM + (n % FLASH_BUCKET_NUM == 0 ? 0 : 1);
  // each block process one row
  if (global_block_id >= group_row_num) {
    return;
  }

  const int row_a = group_row_ptr[global_block_id];

  __shared__ int block_lds_begins[FLASH_BUCKET_NUM];
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
    block_lds_nexts[idx] = HASH_MAP_NULL_VALUE;
  }
  for (int i = block_thread_id; i < FLASH_BUCKET_NUM; i += block_thread_num) {
    block_lds_begins[i] = HASH_MAP_NULL_VALUE;
  }
  if (block_thread_id == 0) {
    block_lds_use_size = 0;
  }

  const I row_idx_begin = row_ptr_a[row_a];
  const I row_idx_end = row_ptr_a[row_a + 1];

  I *const block_mem_ids = ids + row_ptr_c[row_a];
  T *const block_mem_values = values + row_ptr_c[row_a];
#ifdef CAS_PRA_VERIFY
  I *const last_mem_id = ids + row_ptr_c[row_a + 1] - 1;
  T *const last_mem_val = values + row_ptr_c[row_a + 1] - 1;
#endif

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
      linear_chain_hashmap_vector_insert_add<T, I, HASH_MAP_NULL_VALUE>(block_lds_begins, block_lds_nexts,
                                                                        block_lds_ids, block_lds_values, hash_val,
                                                                        col_b, tmp_val, &block_lds_use_size);
    }
    {
      const int r = BLOCK_ROUNDS - 1;
      const int idx_b = idx_begin_b + THREADS * r + block_thread_id;
      if (idx_b < idx_end_b) {
        const T tmp_val = val_A * values_ptr_b[idx_b];
        const I col_b = col_idx_ptr_b[idx_b];
        const size_t hash_val = col_b / FLASH_SORT_WIDTH;
        // insert to lds hashmap
        linear_chain_hashmap_vector_insert_add<T, I, HASH_MAP_NULL_VALUE>(block_lds_begins, block_lds_nexts,
                                                                          block_lds_ids, block_lds_values, hash_val,
                                                                          col_b, tmp_val, &block_lds_use_size);
      }
    }
  }
  __syncthreads();
  for (int i = block_thread_id; i < FLASH_BUCKET_NUM; i += block_thread_num) {
    int count = 0;
    for (int l = block_lds_begins[i]; l != HASH_MAP_NULL_VALUE; l = block_lds_nexts[l]) {
      count++;
    }
    flash_bucket_use_size[i] = count;
  }
  __syncthreads();
  // scan
  if (block_wf_id == 0) {
    const int idx = wf_thread_id;
    int scan_val = flash_bucket_use_size[idx];
    common::scan_in_vector_device<int, __WF_SIZE__>(&scan_val);
    flash_bucket_use_size[idx] = scan_val;
  }
  __syncthreads();
  // flash sort
  const int end_offset = block_lds_use_size;
  for (int i = block_thread_id; i < end_offset; i += block_thread_num) {
    const I base_id = block_lds_ids[i];
    const T base_values = block_lds_values[i];
    const size_t hash_val = base_id / FLASH_SORT_WIDTH;
    int smaller_count = hash_val > 0 ? flash_bucket_use_size[hash_val - 1] : 0;
    for (int l = block_lds_begins[hash_val]; l != HASH_MAP_NULL_VALUE; l = block_lds_nexts[l]) {
      if (block_lds_ids[l] < base_id) {
        smaller_count++;
      }
    }
#ifdef CAS_PRA_VERIFY
    if (smaller_count == end_offset - 1) {
      *last_mem_id = base_id;
      *last_mem_val = base_values * alpha;
      continue;
    }
#endif
    block_mem_ids[smaller_count] = base_id;
    block_mem_values[smaller_count] = base_values * alpha;
  }
}

template <typename T, typename I, size_t N, size_t THREADS, size_t ITEMS_PER_BUCKET>
__global__ void __launch_bounds__(THREADS) spgemm_numeric_block_per_row_flash_sorted_one_pass_kernel(
    const T alpha, const size_t m, const size_t n, const I *row_ptr_a, const I *col_idx_ptr_a, const T *values_ptr_a,
    const I *row_ptr_b, const I *col_idx_ptr_b, const T *values_ptr_b, const I *group_row_ptr, const int group_row_num,
    const I *row_ptr_c, I *ids, T *values, FlashSortMultipleScanTag) {
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
  constexpr int HASH_MAP_NULL_VALUE = 0xffffffff;

  constexpr int FLASH_BUCKET_NUM = LDS_MAP_CAP / ITEMS_PER_BUCKET + (LDS_MAP_CAP % ITEMS_PER_BUCKET == 0 ? 0 : 1);
  const int FLASH_SORT_WIDTH = n / FLASH_BUCKET_NUM + (n % FLASH_BUCKET_NUM == 0 ? 0 : 1);
  // each block process one row
  if (global_block_id >= group_row_num) {
    return;
  }

  const int row_a = group_row_ptr[global_block_id];

  __shared__ int block_lds_begins[FLASH_BUCKET_NUM];
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
    block_lds_nexts[idx] = HASH_MAP_NULL_VALUE;
  }
  for (int i = block_thread_id; i < FLASH_BUCKET_NUM; i += block_thread_num) {
    block_lds_begins[i] = HASH_MAP_NULL_VALUE;
  }
  if (block_thread_id == 0) {
    block_lds_use_size = 0;
  }

  const I row_idx_begin = row_ptr_a[row_a];
  const I row_idx_end = row_ptr_a[row_a + 1];

  I *const block_mem_ids = ids + row_ptr_c[row_a];
  T *const block_mem_values = values + row_ptr_c[row_a];
#ifdef CAS_PRA_VERIFY
  I *const last_mem_id = ids + row_ptr_c[row_a + 1] - 1;
  T *const last_mem_val = values + row_ptr_c[row_a + 1] - 1;
#endif

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
      linear_chain_hashmap_vector_insert_add<T, I, HASH_MAP_NULL_VALUE>(block_lds_begins, block_lds_nexts,
                                                                        block_lds_ids, block_lds_values, hash_val,
                                                                        col_b, tmp_val, &block_lds_use_size);
    }
    {
      const int r = BLOCK_ROUNDS - 1;
      const int idx_b = idx_begin_b + THREADS * r + block_thread_id;
      if (idx_b < idx_end_b) {
        const T tmp_val = val_A * values_ptr_b[idx_b];
        const I col_b = col_idx_ptr_b[idx_b];
        const size_t hash_val = col_b / FLASH_SORT_WIDTH;
        // insert to lds hashmap
        linear_chain_hashmap_vector_insert_add<T, I, HASH_MAP_NULL_VALUE>(block_lds_begins, block_lds_nexts,
                                                                          block_lds_ids, block_lds_values, hash_val,
                                                                          col_b, tmp_val, &block_lds_use_size);
      }
    }
  }
  __syncthreads();
  for (int i = block_thread_id; i < FLASH_BUCKET_NUM; i += block_thread_num) {
    int count = 0;
    for (int l = block_lds_begins[i]; l != HASH_MAP_NULL_VALUE; l = block_lds_nexts[l]) {
      count++;
    }
    flash_bucket_use_size[i] = count;
  }
  __syncthreads();
  // scan
  if (block_wf_id == 0) {
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
  __syncthreads();
  // flash sort
  const int end_offset = block_lds_use_size;
  for (int i = block_thread_id; i < end_offset; i += block_thread_num) {
    const I base_id = block_lds_ids[i];
    const T base_values = block_lds_values[i];
    const size_t hash_val = base_id / FLASH_SORT_WIDTH;
    int smaller_count = hash_val > 0 ? flash_bucket_use_size[hash_val - 1] : 0;
    for (int l = block_lds_begins[hash_val]; l != HASH_MAP_NULL_VALUE; l = block_lds_nexts[l]) {
      if (block_lds_ids[l] < base_id) {
        smaller_count++;
      }
    }
#ifdef CAS_PRA_VERIFY
    if (smaller_count == end_offset - 1) {
      *last_mem_id = base_id;
      *last_mem_val = base_values * alpha;
      continue;
    }
#endif
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
  constexpr int HASH_MAP_NULL_VALUE = 0xffffffff;

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
    vec_lds_begins[idx] = HASH_MAP_NULL_VALUE;
    vec_lds_nexts[idx] = HASH_MAP_NULL_VALUE;
  }
  if (vec_thread_id == 0) {
    *vec_lds_use_size = 0;
  }
  // each vec process one row
  const I row_idx_begin = row_ptr_a[row_a];
  const I row_idx_end = row_ptr_a[row_a + 1];
  I *vec_mem_ids = ids + row_ptr_c[row_a];
  T *vec_mem_values = values + row_ptr_c[row_a];
#ifdef CAS_PRA_VERIFY
  I *last_mem_id = ids + row_ptr_c[row_a + 1] - 1;
  T *last_mem_val = values + row_ptr_c[row_a + 1] - 1;
#endif

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
      linear_chain_hashmap_vector_insert_add<T, I, HASH_MAP_NULL_VALUE, LDS_HASH_MASK>(
          vec_lds_begins, vec_lds_nexts, vec_lds_ids, vec_lds_values, col_b, tmp_val, vec_lds_use_size);
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
#ifdef CAS_PRA_VERIFY
    if (smaller_count == end_offset - 1) {
      *last_mem_id = base_id;
      *last_mem_val = base_values * alpha;
      continue;
    }
#endif
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
  constexpr int HASH_MAP_NULL_VALUE = 0xffffffff;

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
    block_lds_begins[idx] = HASH_MAP_NULL_VALUE;
    block_lds_nexts[idx] = HASH_MAP_NULL_VALUE;
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
      linear_chain_hashmap_vector_insert_add<T, I, HASH_MAP_NULL_VALUE, LDS_HASH_MASK>(
          block_lds_begins, block_lds_nexts, block_lds_ids, block_lds_values, col_b, tmp_val, &block_lds_use_size);
    }
    {
      const int r = BLOCK_ROUNDS - 1;
      const int idx_b = idx_begin_b + THREADS * r + block_thread_id;
      if (idx_b < idx_end_b) {
        const T tmp_val = val_A * values_ptr_b[idx_b];
        const I col_b = col_idx_ptr_b[idx_b];
        // insert to lds hashmap
        linear_chain_hashmap_vector_insert_add<T, I, HASH_MAP_NULL_VALUE, LDS_HASH_MASK>(
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
__global__ void __launch_bounds__(THREADS)
    spgemm_numeric_esc_vector_per_row_sorted_one_pass_kernel(const T alpha, const size_t m, const size_t n,
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
  constexpr size_t CHUNK_CAP_PER_VEC = VEC_SIZE * N;
  constexpr size_t CHUNK_CAP = CHUNK_CAP_PER_VEC * block_vec_num;
  constexpr I CHUNK_NULL_VALUE = 0xffffffff;

  __shared__ I shared_ids[CHUNK_CAP];
  __shared__ T shared_values[CHUNK_CAP];

  I *vec_lds_ids = shared_ids + CHUNK_CAP_PER_VEC * block_vec_id;
  T *vec_lds_values = shared_values + CHUNK_CAP_PER_VEC * block_vec_id;

  // init lds hashmap
#pragma unroll N
  for (int i = 0; i < N; ++i) {
    const int idx = vec_thread_id + vec_thread_num * i;
    vec_lds_ids[idx] = CHUNK_NULL_VALUE;
  }
  // each vec process one row
  const I row_idx_begin = row_ptr_a[row_a];
  const I row_idx_end = row_ptr_a[row_a + 1];
  I *vec_mem_ids = ids + row_ptr_c[row_a];
  T *vec_mem_values = values + row_ptr_c[row_a];
#ifdef CAS_PRA_VERIFY
  int last_item_offset = row_ptr_c[row_a + 1] - row_ptr_c[row_a] - 1;
#endif

  // expand
  int end_offset = 0;
  for (int idx_a = row_idx_begin; idx_a < row_idx_end; ++idx_a) {
    const I col_a = col_idx_ptr_a[idx_a];
    const T val_A = values_ptr_a[idx_a];

    const I row_b = col_a;
    const I idx_begin_b = row_ptr_b[row_b];
    const I idx_end_b = row_ptr_b[row_b + 1];
    for (int idx_b = idx_begin_b + vec_thread_id; idx_b < idx_end_b; idx_b += vec_thread_num) {
      const T tmp_val = val_A * values_ptr_b[idx_b];
      const I col_b = col_idx_ptr_b[idx_b];
      vec_lds_ids[end_offset + idx_b - idx_begin_b] = col_b;
      vec_lds_values[end_offset + idx_b - idx_begin_b] = tmp_val;
    }
    end_offset += idx_end_b - idx_begin_b;
  }
  // select sort
  I reg_id[N];
  T reg_val[N];
  int reg_count[N];
#pragma unroll N
  for (int i = 0; i < N; i++) {
    const int idx = vec_thread_num * i + vec_thread_id;
    if (idx < end_offset) {
      reg_id[i] = vec_lds_ids[idx];
      reg_val[i] = vec_lds_values[idx];
      reg_count[i] = 0;
      for (int j = 0; j < end_offset; ++j) {
        if (vec_lds_ids[j] < reg_id[i]) {
          reg_count[i]++;
        }
      }
    } else {
      reg_count[i] = -1;
    }
  }
  // combine
#pragma unroll N
  for (int i = 0; i < N; ++i) {
    const int idx = vec_thread_num * i + vec_thread_id;
    vec_lds_ids[idx] = CHUNK_NULL_VALUE;
    vec_lds_values[idx] = static_cast<T>(0);
  }
#pragma unroll N
  for (int i = 0; i < N; ++i) {
    if (reg_count[i] != -1) {
      const int idx = reg_count[i];
      vec_lds_ids[idx] = reg_id[i];
      atomicAdd(&vec_lds_values[idx], reg_val[i]);
    }
  }
#ifdef CAS_PRA_VERIFY
  if (vec_thread_id == 0) {
    if (last_item_offset > 0 && vec_lds_ids[last_item_offset] == CHUNK_NULL_VALUE) {
      int idx = last_item_offset - 1;
      while (vec_lds_ids[idx] == CHUNK_NULL_VALUE) {
        idx--;
      }
      // swap item
      vec_lds_ids[last_item_offset] = vec_lds_ids[idx];
      vec_lds_values[last_item_offset] = vec_lds_values[idx];

      vec_lds_ids[idx] = CHUNK_NULL_VALUE;
    }
  }
#endif

#ifndef CAS_PRA_VERIFY
  // notes: if have same items in array, middle of array will have CHUNK_NULL_VALUE
  // scan and re-map
  int scan_sum = 0;
#pragma unroll N
  for (int i = 0; i < N; ++i) {
    const int idx = vec_thread_num * i + vec_thread_id;
    const I base_id = vec_lds_ids[idx];
    const T base_values = vec_lds_values[idx];
    vec_lds_ids[idx] = CHUNK_NULL_VALUE;
    int scan_val = (base_id == CHUNK_NULL_VALUE ? 1 : 0);
    common::scan_in_vector_device<int, VEC_SIZE>(&scan_val);
    scan_val += scan_sum;
    if (base_id != CHUNK_NULL_VALUE) {
      vec_lds_ids[idx - scan_val] = base_id;
      vec_lds_values[idx - scan_val] = base_values;
    }
    scan_sum += __shfl(scan_val, VEC_SIZE - 1, VEC_SIZE);
  }
#endif
  // write back
#pragma unroll N
  for (int i = 0; i < N; ++i) {
    const int idx = vec_thread_id + vec_thread_num * i;
    if (vec_lds_ids[idx] != CHUNK_NULL_VALUE) {
      vec_mem_ids[idx] = vec_lds_ids[idx];
      vec_mem_values[idx] = vec_lds_values[idx] * alpha;
    }
  }
}

template <typename T, typename I, size_t N, size_t THREADS>
__global__ void __launch_bounds__(THREADS)
    spgemm_numeric_esc_block_per_row_sorted_one_pass_kernel(const T alpha, const size_t m, const size_t n,
                                                            const I *row_ptr_a, const I *col_idx_ptr_a,
                                                            const T *values_ptr_a, const I *row_ptr_b,
                                                            const I *col_idx_ptr_b, const T *values_ptr_b,
                                                            const I *group_row_ptr, const int group_row_num,
                                                            const I *row_ptr_c, I *ids, T *values) {
  // block id in global
  const int global_block_id = blockIdx.x;
  // block num in global
  const int global_block_num = gridDim.x;
  // thread id in block
  const int block_thread_id = threadIdx.x;
  // thread num in block
  constexpr int block_thread_num = THREADS;

  // each block process one row
  if (global_block_id >= group_row_num) {
    return;
  }
  const int row_a = group_row_ptr[global_block_id];

  // hashmap in LDS
  constexpr size_t CHUNK_CAP_PER_BLOCK = THREADS * N;
  constexpr size_t CHUNK_CAP = CHUNK_CAP_PER_BLOCK;
  constexpr I CHUNK_NULL_VALUE = 0xffffffff;

  __shared__ I block_lds_ids[CHUNK_CAP];
  __shared__ T block_lds_values[CHUNK_CAP];

  // init lds hashmap
#pragma unroll N
  for (int i = 0; i < N; ++i) {
    const int idx = block_thread_id + block_thread_num * i;
    block_lds_ids[idx] = CHUNK_NULL_VALUE;
  }
  __syncthreads();
  // each block process one row
  const I row_idx_begin = row_ptr_a[row_a];
  const I row_idx_end = row_ptr_a[row_a + 1];
  I *block_mem_ids = ids + row_ptr_c[row_a];
  T *block_mem_values = values + row_ptr_c[row_a];
#ifdef CAS_PRA_VERIFY
  int last_item_offset = row_ptr_c[row_a + 1] - row_ptr_c[row_a] - 1;
#endif

  // expand
  int end_offset = 0;
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
      block_lds_ids[end_offset + idx_b - idx_begin_b] = col_b;
      block_lds_values[end_offset + idx_b - idx_begin_b] = tmp_val;
    }
    {
      const int r = BLOCK_ROUNDS - 1;
      const int idx_b = idx_begin_b + THREADS * r + block_thread_id;
      if (idx_b < idx_end_b) {
        const T tmp_val = val_A * values_ptr_b[idx_b];
        const I col_b = col_idx_ptr_b[idx_b];
        block_lds_ids[end_offset + idx_b - idx_begin_b] = col_b;
        block_lds_values[end_offset + idx_b - idx_begin_b] = tmp_val;
      }
    }
    end_offset += idx_end_b - idx_begin_b;
  }
  __syncthreads();
  // select sort
  I reg_id[N];
  T reg_val[N];
  int reg_count[N];
#pragma unroll N
  for (int i = 0; i < N; i++) {
    const int idx = block_thread_num * i + block_thread_id;
    if (idx < end_offset) {
      reg_id[i] = block_lds_ids[idx];
      reg_val[i] = block_lds_values[idx];
      reg_count[i] = 0;
      for (int j = 0; j < end_offset; ++j) {
        if (block_lds_ids[j] < reg_id[i]) {
          reg_count[i]++;
        }
      }
    } else {
      reg_count[i] = -1;
    }
  }
  __syncthreads();
  // combine
#pragma unroll N
  for (int i = 0; i < N; ++i) {
    const int idx = block_thread_num * i + block_thread_id;
    block_lds_ids[idx] = CHUNK_NULL_VALUE;
    block_lds_values[idx] = static_cast<T>(0);
  }
  __syncthreads();
#pragma unroll N
  for (int i = 0; i < N; ++i) {
    if (reg_count[i] != -1) {
      const int idx = reg_count[i];
      block_lds_ids[idx] = reg_id[i];
      atomicAdd(&block_lds_values[idx], reg_val[i]);
    }
  }
  __syncthreads();
#ifdef CAS_PRA_VERIFY
  if (block_thread_id == 0) {
    if (last_item_offset > 0 && block_lds_ids[last_item_offset] == CHUNK_NULL_VALUE) {
      int idx = last_item_offset - 1;
      while (block_lds_ids[idx] == CHUNK_NULL_VALUE) {
        idx--;
      }
      // swap item
      block_lds_ids[last_item_offset] = block_lds_ids[idx];
      block_lds_values[last_item_offset] = block_lds_values[idx];

      block_lds_ids[idx] = CHUNK_NULL_VALUE;
    }
  }
  __syncthreads();
#endif
#ifndef CAS_PRA_VERIFY
  // todo: fix it, need scan and re-map
#endif
  // write back
#pragma unroll N
  for (int i = 0; i < N; ++i) {
    const int idx = block_thread_id + block_thread_num * i;
    if (block_lds_ids[idx] != CHUNK_NULL_VALUE) {
      block_mem_ids[idx] = block_lds_ids[idx];
      block_mem_values[idx] = block_lds_values[idx] * alpha;
    }
  }
}

#endif // SPGEMM_KOKKOS_SPGEMM_NUMERIC_KERNEL_H
