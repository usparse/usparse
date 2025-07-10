#ifndef USTB_SPARSE_SPGEMM_SPGEMM_NUMERIC_ESC_KERNEL_HPP
#define USTB_SPARSE_SPGEMM_SPGEMM_NUMERIC_ESC_KERNEL_HPP

#include "spgemm_common/scan/scan_in_device.hpp"

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

  // each vec process one row
  const I row_idx_begin = row_ptr_a[row_a];
  const I row_idx_end = row_ptr_a[row_a + 1];
  I *vec_mem_ids = ids + row_ptr_c[row_a];
  T *vec_mem_values = values + row_ptr_c[row_a];

  int last_item_offset = row_ptr_c[row_a + 1] - row_ptr_c[row_a] - 1;

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

  // write back
  for (int idx = vec_thread_id; idx <= last_item_offset; idx += vec_thread_num) {
    if (vec_lds_ids[idx] != CHUNK_NULL_VALUE) {
      vec_mem_ids[idx] = vec_lds_ids[idx];
      vec_mem_values[idx] = vec_lds_values[idx] * alpha;
    } else {
      vec_mem_ids[idx] = static_cast<I>(0);
      vec_mem_values[idx] = static_cast<T>(0);
    }
  }
}

template <typename T, typename I, size_t N, size_t THREADS, size_t ITEMS_PER_BUCKET>
__global__ void __launch_bounds__(THREADS) spgemm_numeric_esc_block_per_row_sorted_one_pass_kernel(
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

  constexpr size_t VEC_SIZE = ITEMS_PER_BUCKET;
  // vec id in block
  const int block_vec_id = threadIdx.x / VEC_SIZE;
  // vec num in block
  constexpr int block_vec_num = THREADS / VEC_SIZE;
  // thread id in vec
  const int vec_thread_id = block_thread_id % VEC_SIZE;
  // thread num in vec
  constexpr int vec_thread_num = VEC_SIZE;
  // hashmap in LDS
  constexpr size_t CHUNK_CAP_PER_BLOCK = THREADS * N;
  constexpr size_t CHUNK_CAP = CHUNK_CAP_PER_BLOCK;
  constexpr I CHUNK_NULL_VALUE = 0xffffffff;

  constexpr int FLASH_BUCKET_NUM = CHUNK_CAP / ITEMS_PER_BUCKET + (CHUNK_CAP % ITEMS_PER_BUCKET == 0 ? 0 : 1);
  const int FLASH_SORT_WIDTH = n / FLASH_BUCKET_NUM + (n % FLASH_BUCKET_NUM == 0 ? 0 : 1);
  // each block process one row
  if (global_block_id >= group_row_num) {
    return;
  }
  const int row_a = group_row_ptr[global_block_id];

  __shared__ I block_lds_ids[CHUNK_CAP];
  __shared__ T block_lds_values[CHUNK_CAP];
  __shared__ int block_lds_count[CHUNK_CAP];
  // re-use lds
  I *block_lds_idx_b = block_lds_ids;
  T *block_lds_val_a = block_lds_values;

  constexpr int PADDING_BUCKET_N = FLASH_BUCKET_NUM / __WF_SIZE__ + (FLASH_BUCKET_NUM % __WF_SIZE__ == 0 ? 0 : 1);
  constexpr size_t PADDING_FLASH_BUCKET_NUM = PADDING_BUCKET_N * __WF_SIZE__;
  __shared__ int flash_bucket_use_size[PADDING_FLASH_BUCKET_NUM];

  I reg_id[N];
  T reg_val[N];
  int reg_count[N];
  for (int i = block_thread_id; i < PADDING_FLASH_BUCKET_NUM; i += block_thread_num) {
    flash_bucket_use_size[i] = 0;
  }
  __syncthreads();
  // each block process one row
  const I row_idx_begin = row_ptr_a[row_a];
  const I row_idx_end = row_ptr_a[row_a + 1];
  I *block_mem_ids = ids + row_ptr_c[row_a];
  T *block_mem_values = values + row_ptr_c[row_a];

  int last_item_offset = row_ptr_c[row_a + 1] - row_ptr_c[row_a] - 1;

  // expand
  int end_offset = 0;
  for (int idx_a = row_idx_begin; idx_a < row_idx_end; ++idx_a) {
    const I col_a = col_idx_ptr_a[idx_a];
    const T val_A = values_ptr_a[idx_a];

    const I row_b = col_a;
    const I idx_begin_b = row_ptr_b[row_b];
    const I idx_end_b = row_ptr_b[row_b + 1];
    for (int idx_b = idx_begin_b + block_thread_id; idx_b < idx_end_b; idx_b += THREADS) {
      const int lds_idx = end_offset + idx_b - idx_begin_b;
      block_lds_idx_b[lds_idx] = idx_b;
      block_lds_val_a[lds_idx] = val_A;
    }
    end_offset += idx_end_b - idx_begin_b;
  }
  __syncthreads();
  for (int i = block_thread_id; i < end_offset; i += block_thread_num) {
    const I idx_b = block_lds_idx_b[i];
    const T val_A = block_lds_val_a[i];
    const T tmp_val = val_A * values_ptr_b[idx_b];
    const I col_b = col_idx_ptr_b[idx_b];
    const size_t hash_val = col_b / FLASH_SORT_WIDTH;
    block_lds_ids[i] = col_b;
    block_lds_values[i] = tmp_val;
    block_lds_count[i] = atomicAdd(&flash_bucket_use_size[hash_val], 1);
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
  // remap
#pragma unroll N
  for (int i = 0; i < N; ++i) {
    const int idx = block_thread_id + block_thread_num * i;
    reg_id[i] = block_lds_ids[idx];
    reg_val[i] = block_lds_values[idx];
    reg_count[i] = block_lds_count[idx];
  }
  __syncthreads();
#pragma unroll N
  for (int i = 0; i < N; ++i) {
    const int idx = block_thread_id + block_thread_num * i;
    if (idx < end_offset) {
      const size_t hash_val = reg_id[i] / FLASH_SORT_WIDTH;
      const int remap_idx = (hash_val > 0 ? flash_bucket_use_size[hash_val - 1] : 0) + reg_count[i];
      block_lds_ids[remap_idx] = reg_id[i];
      block_lds_values[remap_idx] = reg_val[i];
    }
  }
  __syncthreads();
  // flash sort
  for (int bucket_id = block_vec_id; bucket_id < FLASH_BUCKET_NUM; bucket_id += block_vec_num) {
    const int idx_begin = bucket_id > 0 ? flash_bucket_use_size[bucket_id - 1] : 0;
    const int idx_end = flash_bucket_use_size[bucket_id];
    for (int idx = idx_begin + vec_thread_id; idx < idx_end; idx += vec_thread_num) {
      const I base_id = block_lds_ids[idx];
      const T base_values = block_lds_values[idx];
      int smaller_count = idx_begin;
      for (int i = idx_begin; i < idx_end; i++) {
        if (block_lds_ids[i] < base_id) {
          smaller_count++;
        }
      }
      block_lds_count[idx] = smaller_count;
    }
  }
  __syncthreads();
#pragma unroll N
  for (int i = 0; i < N; ++i) {
    const int idx = block_thread_id + block_thread_num * i;
    if (idx < end_offset) {
      reg_id[i] = block_lds_ids[idx];
      reg_val[i] = block_lds_values[idx];
      reg_count[i] = block_lds_count[idx];
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

  // write back
  for (int idx = block_thread_id; idx <= last_item_offset; idx += block_thread_num) {
    if (block_lds_ids[idx] != CHUNK_NULL_VALUE) {
      block_mem_ids[idx] = block_lds_ids[idx];
      block_mem_values[idx] = block_lds_values[idx] * alpha;
    } else {
      block_mem_ids[idx] = static_cast<I>(0);
      block_mem_values[idx] = static_cast<T>(0);
    }
  }
}

template <typename T, typename I, size_t N, size_t THREADS, size_t ITEMS_PER_BUCKET>
__global__ void __launch_bounds__(THREADS) spgemm_numeric_esc_block_per_row_sorted_one_pass_kernel(
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

  constexpr size_t VEC_SIZE = ITEMS_PER_BUCKET;
  // vec id in block
  const int block_vec_id = threadIdx.x / VEC_SIZE;
  // vec num in block
  constexpr int block_vec_num = THREADS / VEC_SIZE;
  // thread id in vec
  const int vec_thread_id = block_thread_id % VEC_SIZE;
  // thread num in vec
  constexpr int vec_thread_num = VEC_SIZE;
  // hashmap in LDS
  constexpr size_t CHUNK_CAP_PER_BLOCK = THREADS * N;
  constexpr size_t CHUNK_CAP = CHUNK_CAP_PER_BLOCK;
  constexpr I CHUNK_NULL_VALUE = 0xffffffff;

  constexpr int FLASH_BUCKET_NUM = CHUNK_CAP / ITEMS_PER_BUCKET + (CHUNK_CAP % ITEMS_PER_BUCKET == 0 ? 0 : 1);
  const int FLASH_SORT_WIDTH = n / FLASH_BUCKET_NUM + (n % FLASH_BUCKET_NUM == 0 ? 0 : 1);
  // each block process one row
  if (global_block_id >= group_row_num) {
    return;
  }
  const int row_a = group_row_ptr[global_block_id];

  __shared__ I block_lds_ids[CHUNK_CAP];
  __shared__ T block_lds_values[CHUNK_CAP];
  __shared__ int block_lds_count[CHUNK_CAP];
  // re-use lds
  I *block_lds_idx_b = block_lds_ids;
  T *block_lds_val_a = block_lds_values;

  constexpr int PADDING_BUCKET_N = FLASH_BUCKET_NUM / __WF_SIZE__ + (FLASH_BUCKET_NUM % __WF_SIZE__ == 0 ? 0 : 1);
  constexpr size_t PADDING_FLASH_BUCKET_NUM = PADDING_BUCKET_N * __WF_SIZE__;
  __shared__ int flash_bucket_use_size[PADDING_FLASH_BUCKET_NUM];

  I reg_id[N];
  T reg_val[N];
  int reg_count[N];
  for (int i = block_thread_id; i < PADDING_FLASH_BUCKET_NUM; i += block_thread_num) {
    flash_bucket_use_size[i] = 0;
  }
  __syncthreads();
  // each block process one row
  const I row_idx_begin = row_ptr_a[row_a];
  const I row_idx_end = row_ptr_a[row_a + 1];
  I *block_mem_ids = ids + row_ptr_c[row_a];
  T *block_mem_values = values + row_ptr_c[row_a];

  int last_item_offset = row_ptr_c[row_a + 1] - row_ptr_c[row_a] - 1;

  // expand
  int end_offset = 0;
  for (int idx_a = row_idx_begin; idx_a < row_idx_end; ++idx_a) {
    const I col_a = col_idx_ptr_a[idx_a];
    const T val_A = values_ptr_a[idx_a];

    const I row_b = col_a;
    const I idx_begin_b = row_ptr_b[row_b];
    const I idx_end_b = row_ptr_b[row_b + 1];
    for (int idx_b = idx_begin_b + block_thread_id; idx_b < idx_end_b; idx_b += THREADS) {
      const int lds_idx = end_offset + idx_b - idx_begin_b;
      block_lds_idx_b[lds_idx] = idx_b;
      block_lds_val_a[lds_idx] = val_A;
    }
    end_offset += idx_end_b - idx_begin_b;
  }
  __syncthreads();
  for (int i = block_thread_id; i < end_offset; i += block_thread_num) {
    const I idx_b = block_lds_idx_b[i];
    const T val_A = block_lds_val_a[i];
    const T tmp_val = val_A * values_ptr_b[idx_b];
    const I col_b = col_idx_ptr_b[idx_b];
    const size_t hash_val = col_b / FLASH_SORT_WIDTH;
    block_lds_ids[i] = col_b;
    block_lds_values[i] = tmp_val;
    block_lds_count[i] = atomicAdd(&flash_bucket_use_size[hash_val], 1);
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
  // remap
#pragma unroll N
  for (int i = 0; i < N; ++i) {
    const int idx = block_thread_id + block_thread_num * i;
    reg_id[i] = block_lds_ids[idx];
    reg_val[i] = block_lds_values[idx];
    reg_count[i] = block_lds_count[idx];
  }
  __syncthreads();
#pragma unroll N
  for (int i = 0; i < N; ++i) {
    const int idx = block_thread_id + block_thread_num * i;
    if (idx < end_offset) {
      const size_t hash_val = reg_id[i] / FLASH_SORT_WIDTH;
      const int remap_idx = (hash_val > 0 ? flash_bucket_use_size[hash_val - 1] : 0) + reg_count[i];
      block_lds_ids[remap_idx] = reg_id[i];
      block_lds_values[remap_idx] = reg_val[i];
    }
  }
  __syncthreads();
  // flash sort
  for (int bucket_id = block_vec_id; bucket_id < FLASH_BUCKET_NUM; bucket_id += block_vec_num) {
    const int idx_begin = bucket_id > 0 ? flash_bucket_use_size[bucket_id - 1] : 0;
    const int idx_end = flash_bucket_use_size[bucket_id];
    for (int idx = idx_begin + vec_thread_id; idx < idx_end; idx += vec_thread_num) {
      const I base_id = block_lds_ids[idx];
      const T base_values = block_lds_values[idx];
      int smaller_count = idx_begin;
      for (int i = idx_begin; i < idx_end; i++) {
        if (block_lds_ids[i] < base_id) {
          smaller_count++;
        }
      }
      block_lds_count[idx] = smaller_count;
    }
  }
  __syncthreads();
#pragma unroll N
  for (int i = 0; i < N; ++i) {
    const int idx = block_thread_id + block_thread_num * i;
    if (idx < end_offset) {
      reg_id[i] = block_lds_ids[idx];
      reg_val[i] = block_lds_values[idx];
      reg_count[i] = block_lds_count[idx];
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

  // write back
  for (int idx = block_thread_id; idx <= last_item_offset; idx += block_thread_num) {
    if (block_lds_ids[idx] != CHUNK_NULL_VALUE) {
      block_mem_ids[idx] = block_lds_ids[idx];
      block_mem_values[idx] = block_lds_values[idx] * alpha;
    } else {
      block_mem_ids[idx] = static_cast<I>(0);
      block_mem_values[idx] = static_cast<T>(0);
    }
  }
}

#endif // USTB_SPARSE_SPGEMM_SPGEMM_NUMERIC_ESC_KERNEL_HPP
