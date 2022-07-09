#ifndef SPGEMM_KOKKOS_SPGEMM_KERNEL_HPP
#define SPGEMM_KOKKOS_SPGEMM_KERNEL_HPP

#include "build_flag.h"
#include "hash_map.h"

template <typename T, typename I, size_t BitsLen, size_t WF_SIZE>
__global__ void compress_matrix_kernel(const size_t m, const size_t n, const I *row_ptr, const I *col_idx_ptr, T *arr,
                                       const size_t nums_per_row) {
  // thread id in global
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  // thread num in global
  const int global_thread_num = gridDim.x * blockDim.x;
  // each thread process a row
  for (int row = global_thread_id; row < m; row += global_thread_num) {
    const I row_idx_begin = row_ptr[row];
    const I row_idx_end = row_ptr[row + 1];
    for (int idx = row_idx_begin; idx < row_idx_end; idx++) {
      const I col = col_idx_ptr[idx];
      const int bitset_idx = row * nums_per_row + col / BitsLen;
      arr[bitset_idx] |= 0x1 << (BitsLen - 1 - (col % BitsLen));
    }
  }
}

template <typename T, typename I, size_t BitsLen, size_t WF_SIZE>
__global__ void spgemm_symbolic_kernel(const size_t m, const size_t n, const I *row_ptr, const I *col_idx_ptr,
                                       const T *bitset_b, const size_t nums_per_row, I *row_ptr_c) {
  // thread id in global
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  // thread num in global
  const int global_thread_num = gridDim.x * blockDim.x;
  // each thread process a row
  for (int row = global_thread_id; row < m; row += global_thread_num) {
    const I row_idx_begin = row_ptr[row];
    const I row_idx_end = row_ptr[row + 1];
    I row_count = 0;
    for (int offset = 0; offset < nums_per_row; ++offset) {
      T bits = 0;
      for (int idx = row_idx_begin; idx < row_idx_end; ++idx) {
        const I col = col_idx_ptr[idx];
        bits |= bitset_b[col * nums_per_row + offset];
      }
      while (bits != 0) {
        if (bits & 0x1) {
          row_count++;
        }
        bits = bits >> 1;
      }
    }
    row_ptr_c[row + 1] = row_count;
  }
}

template <typename T, typename I, size_t N, size_t LDS_MAP_CAP, size_t LDS_HASH_MASK, size_t BLOCKS, size_t WF_SIZE>
__global__ void spgemm_numeric_kernel(const T alpha, const size_t m, const size_t n, const I *row_ptr_a,
                                      const I *col_idx_ptr_a, const T *values_ptr_a, const I *row_ptr_b,
                                      const I *col_idx_ptr_b, const T *values_ptr_b, const I *row_ptr_c, int *begins,
                                      int *nexts, I *ids, T *values) {
  // thread id in global
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  // thread num in global
  const int global_thread_num = gridDim.x * blockDim.x;
  // wf id in global
  const int global_wf_id = global_thread_id / WF_SIZE;
  // wf num in global
  const int global_wf_num = global_thread_num / WF_SIZE;
  // block id in global
  const int global_block_id = blockIdx.x;
  // thread id in block
  const int block_thread_id = threadIdx.x;
  // thread num in block
  constexpr int block_thread_num = BLOCKS;
  // wf id in block
  const int block_wf_id = threadIdx.x / WF_SIZE;
  // wf num in block
  constexpr int block_wf_num = BLOCKS / WF_SIZE;
  // thread id in wf
  const int wf_thread_id = global_thread_id % WF_SIZE;
  // thread num in wf
  constexpr int wf_thread_num = WF_SIZE;
  // hashmap in LDS
  constexpr int LDS_BEGINS_SIZE = LDS_MAP_CAP * block_wf_num;
#ifndef GLOBAL_LOAD_X2
  constexpr int WF_PROCESS_ONCE = WF_SIZE;
#endif
#ifdef GLOBAL_LOAD_X2
  constexpr int WF_PROCESS_ONCE = WF_SIZE * 2;
#endif
  constexpr int NEXTS_SIZE_PER_WF = WF_PROCESS_ONCE * N;
  constexpr int LDS_NEXTS_SIZE = NEXTS_SIZE_PER_WF * block_wf_num;
  __shared__ int shared_begins[LDS_BEGINS_SIZE];
  __shared__ int shared_nexts[LDS_NEXTS_SIZE];
  __shared__ I shared_ids[LDS_NEXTS_SIZE];
  __shared__ T shared_values[LDS_NEXTS_SIZE];
  // init lds hashmap
  for (int i = block_thread_id; i < LDS_BEGINS_SIZE; i += block_thread_num) {
    shared_begins[i] = HashMap<T, I>::NullValue;
  }
  for (int i = block_thread_id; i < LDS_NEXTS_SIZE; i += block_thread_num) {
    shared_nexts[i] = HashMap<T, I>::NullValue;
  }
  // each wf process one row
  __shared__ int lds_nexts_offset[block_wf_num];
  __shared__ int mem_nexts_offset[block_wf_num];

  for (int row_a = global_wf_id; row_a < m; row_a += global_wf_num) {
    const I row_idx_begin = row_ptr_a[row_a];
    const I row_idx_end = row_ptr_a[row_a + 1];
    const size_t lds_begins_start = LDS_MAP_CAP * block_wf_id;
    const size_t lds_nexts_start = NEXTS_SIZE_PER_WF * block_wf_id;
    const size_t lds_nexts_end = lds_nexts_start + NEXTS_SIZE_PER_WF;

    const size_t mem_begins_start = HashMap<T, I>::Capacity * row_a;
    const size_t mem_nexts_start = row_ptr_c[row_a];
    const size_t mem_nexts_end = row_ptr_c[row_a + 1];
    if (wf_thread_id == 0) {
      lds_nexts_offset[block_wf_id] = 0;
      mem_nexts_offset[block_wf_id] = 0;
    }
    for (int idx_a = row_idx_begin; idx_a < row_idx_end; ++idx_a) {
      const I col_a = col_idx_ptr_a[idx_a];
      const I row_b = col_a;
      const I idx_begin_b = row_ptr_b[row_b];
      const I idx_end_b = row_ptr_b[row_b + 1];
      const int wf_rounds =
          (idx_end_b - idx_begin_b) / WF_PROCESS_ONCE + ((idx_end_b - idx_begin_b) % WF_PROCESS_ONCE == 0 ? 0 : 1);
      for (int round = 0; round < wf_rounds; ++round) {
        if (lds_nexts_offset[block_wf_id] + WF_PROCESS_ONCE > NEXTS_SIZE_PER_WF) {
          // if lds hashmap is full, write to device memory hashmap
          const int end_offset = lds_nexts_offset[block_wf_id];
          for (int i = lds_nexts_start + wf_thread_id; i < lds_nexts_start + end_offset; i += wf_thread_num) {
            hashmap_wavefront_insert<T, I, HashMap<T, I>::HashMask>(begins, nexts, ids, values, mem_nexts_offset,
                                                                    block_wf_id, mem_begins_start, mem_nexts_start,
                                                                    shared_ids[i], shared_values[i]);
          }
          // reset lds hashmap
          for (int i = lds_begins_start + wf_thread_id; i < lds_begins_start + LDS_MAP_CAP; i += wf_thread_num) {
            shared_begins[i] = HashMap<T, I>::NullValue;
          }
          for (int i = lds_nexts_start + wf_thread_id; i < lds_nexts_start + end_offset; i += wf_thread_num) {
            shared_nexts[i] = HashMap<T, I>::NullValue;
          }
          if (wf_thread_id == 0) {
            lds_nexts_offset[block_wf_id] = 0;
          }
        }
#ifndef GLOBAL_LOAD_X2
        const int idx_b = idx_begin_b + WF_PROCESS_ONCE * round + wf_thread_id;
        if (idx_b < idx_end_b) {
          const T tmp_val = values_ptr_a[idx_a] * values_ptr_b[idx_b];
          const I col_b = col_idx_ptr_b[idx_b];
          // insert to lds hashmap
          hashmap_wavefront_insert<T, I, LDS_HASH_MASK>(shared_begins, shared_nexts, shared_ids, shared_values,
                                                        lds_nexts_offset, block_wf_id, lds_begins_start,
                                                        lds_nexts_start, col_b, tmp_val);
        }
#endif
#ifdef GLOBAL_LOAD_X2
        const int idx_b = idx_begin_b + WF_PROCESS_ONCE * round + wf_thread_id * 2;
        if (idx_b < idx_end_b) {
          if (idx_b + 1 < idx_end_b) {
            const T tmp_val_0 = values_ptr_a[idx_a] * values_ptr_b[idx_b];
            const I col_b_0 = col_idx_ptr_b[idx_b];
            const T tmp_val_1 = values_ptr_a[idx_a] * values_ptr_b[idx_b + 1];
            const I col_b_1 = col_idx_ptr_b[idx_b + 1];
            // insert to lds hashmap
            hashmap_wavefront_insert<T, I, LDS_HASH_MASK>(shared_begins, shared_nexts, shared_ids, shared_values,
                                                          lds_nexts_offset, block_wf_id, lds_begins_start,
                                                          lds_nexts_start, col_b_0, tmp_val_0);
            hashmap_wavefront_insert<T, I, LDS_HASH_MASK>(shared_begins, shared_nexts, shared_ids, shared_values,
                                                          lds_nexts_offset, block_wf_id, lds_begins_start,
                                                          lds_nexts_start, col_b_1, tmp_val_1);
          } else {
            const T tmp_val = values_ptr_a[idx_a] * values_ptr_b[idx_b];
            const I col_b = col_idx_ptr_b[idx_b];
            // insert to lds hashmap
            hashmap_wavefront_insert<T, I, LDS_HASH_MASK>(shared_begins, shared_nexts, shared_ids, shared_values,
                                                          lds_nexts_offset, block_wf_id, lds_begins_start,
                                                          lds_nexts_start, col_b, tmp_val);
          }
        }
#endif
      }
      // write to device memory hashmap
      const int end_offset = lds_nexts_offset[block_wf_id];
      for (int i = lds_nexts_start + wf_thread_id; i < lds_nexts_start + end_offset; i += wf_thread_num) {
        hashmap_wavefront_insert<T, I, HashMap<T, I>::HashMask>(begins, nexts, ids, values, mem_nexts_offset,
                                                                block_wf_id, mem_begins_start, mem_nexts_start,
                                                                shared_ids[i], shared_values[i]);
      }
      // reset lds hashmap
      for (int i = lds_begins_start + wf_thread_id; i < lds_begins_start + LDS_MAP_CAP; i += wf_thread_num) {
        shared_begins[i] = HashMap<T, I>::NullValue;
      }
      for (int i = lds_nexts_start + wf_thread_id; i < lds_nexts_start + end_offset; i += wf_thread_num) {
        shared_nexts[i] = HashMap<T, I>::NullValue;
      }
      if (wf_thread_id == 0) {
        lds_nexts_offset[block_wf_id] = 0;
      }
    }
    for (int values_idx = mem_nexts_start + wf_thread_id; values_idx < mem_nexts_end; values_idx += wf_thread_num) {
      values[values_idx] *= alpha;
    }
  }
}

#endif // SPGEMM_KOKKOS_SPGEMM_KERNEL_HPP
