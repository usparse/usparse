//
// Created by chu genshen on 2021/9/8.
//

#ifndef USPARSE_HIP_USPARSE_FLAT_KERNELS_IMP_REDUCE_HPP
#define USPARSE_HIP_USPARSE_FLAT_KERNELS_IMP_REDUCE_HPP

#include "common/utils.h"
#include "usparse_types.h"
#include "flat_mtx_info.h"
#include "primitives/reduce/wavefront/wavefront_shlf_utils.hpp"

template <typename T> __device__ __forceinline__ void flat_atomic_add(T *address, T val) { atomicAdd(address, val); }

// Complex number specialized version: Single precision
__device__ __forceinline__ void flat_atomic_add(usparse::usparse_fpc32 *address, usparse::usparse_fpc32 val) {
  float *float_addr = reinterpret_cast<float *>(address);
  atomicAdd(float_addr, std::real(val));
  atomicAdd(float_addr + 1, std::imag(val));
  return;
}

// Complex number specialized version: double precision
__device__ __forceinline__ void flat_atomic_add(usparse::usparse_fpc64 *address, usparse::usparse_fpc64 val) {
  double *double_addr = reinterpret_cast<double *>(address);
  atomicAdd(double_addr, std::real(val));
  atomicAdd(double_addr + 1, std::imag(val));
  return;
}

template <typename BP_INDEX_TYPE, typename SHARE_MEM_DATA_TYPE, typename SHARE_MEM_INDEX_TYPE, typename NNZ_TYPE,
          typename ROW_TYPE, typename SCALE_DATA_TYPE, typename COMPUTE_TYPE, typename Y_TYPE, int NNZ_PER_BLOCK, int R,
          int THREADS>
__device__ __forceinline__ void
flat_reduce_segment_sum(const int tid_in_block, const int bp_index, const ROW_TYPE reduce_start_row_id,
                        const ROW_TYPE reduce_end_row_id, const SCALE_DATA_TYPE alpha,
                        const NNZ_TYPE *__restrict__ row_offset, const NNZ_TYPE last_element_index,
                        ROW_TYPE *__restrict__ _lds_shared_idx, SHARE_MEM_DATA_TYPE *__restrict__ _lds_shared_data,
                        Y_TYPE *__restrict__ y) {
  ROW_TYPE row = reduce_start_row_id;
#pragma unroll
  for (int i = 0; i < R; ++i) {
    SHARE_MEM_DATA_TYPE *_lds_shared_data_cur = _lds_shared_data + i * THREADS;
    const NNZ_TYPE cur_idx = bp_index * NNZ_PER_BLOCK + i * THREADS + tid_in_block;
    for (ROW_TYPE j = row + 1; j <= reduce_end_row_id; ++j) {
      if (row_offset[j] - cur_idx > 0) {
        row = j - 1;
        _lds_shared_idx[tid_in_block] = row;
        break;
      }
    }
    __syncthreads();
    block_segment_sum<ROW_TYPE, SHARE_MEM_DATA_TYPE, THREADS>(_lds_shared_idx, _lds_shared_data_cur);
    if (cur_idx < last_element_index &&
        (tid_in_block == THREADS - 1 || _lds_shared_idx[tid_in_block] != _lds_shared_idx[tid_in_block + 1] ||
         cur_idx == last_element_index - 1)) {
      flat_atomic_add(&y[_lds_shared_idx[tid_in_block]],
                      static_cast<Y_TYPE>(static_cast<COMPUTE_TYPE>(alpha) *
                                          static_cast<COMPUTE_TYPE>(_lds_shared_data_cur[tid_in_block])));
    }
    __syncthreads();
  };
}

template <typename BP_INDEX_TYPE, typename SHARE_MEM_DATA_TYPE, typename SHARE_MEM_INDEX_TYPE, typename NNZ_TYPE,
          typename ROW_TYPE, typename SCALE_DATA_TYPE, typename COMPUTE_TYPE, typename Y_TYPE, int NNZ_PER_BLOCK,
          int THREADS, int VECTOR_SIZE>
__device__ __forceinline__ void
flat_reduce_in_vector(const ROW_TYPE n_reduce_rows_num, const int tid_in_block, const BP_INDEX_TYPE bp_index,
                      const ROW_TYPE reduce_start_row_id, const ROW_TYPE reduce_end_row_id, const SCALE_DATA_TYPE alpha,
                      const NNZ_TYPE *__restrict__ row_offset, const SHARE_MEM_DATA_TYPE *__restrict__ _lds_shared_data,
                      Y_TYPE *__restrict__ y) {
  // use `vec_num` vectors, each vector can process reduction of one row by involving `vec_size` threads.
  const int vec_size = VECTOR_SIZE;
  const int vec_num = THREADS / vec_size;
  const int vec_id = tid_in_block / vec_size;
  const int tid_in_vec = tid_in_block % vec_size;

  ROW_TYPE reduce_row_id = reduce_start_row_id + vec_id;
  if (reduce_row_id < reduce_end_row_id) {
    const SHARE_MEM_INDEX_TYPE reduce_start_inx =
        max(static_cast<SHARE_MEM_INDEX_TYPE>(0),
            static_cast<SHARE_MEM_INDEX_TYPE>(row_offset[reduce_row_id] - bp_index * NNZ_PER_BLOCK));
    const SHARE_MEM_INDEX_TYPE reduce_end_inx =
        min(static_cast<SHARE_MEM_INDEX_TYPE>(NNZ_PER_BLOCK),
            static_cast<SHARE_MEM_INDEX_TYPE>(row_offset[reduce_row_id + 1] - bp_index * NNZ_PER_BLOCK));
    // reduce LDS via vectors.
    COMPUTE_TYPE sum = static_cast<COMPUTE_TYPE>(0);
    for (SHARE_MEM_INDEX_TYPE i = reduce_start_inx + tid_in_vec; i < reduce_end_inx; i += vec_size) {
      sum += _lds_shared_data[i];
    }
    // reduce in vector
    for (int i = vec_size >> 1; i > 0; i >>= 1) {
      sum += __shfl_down(sum, i, vec_size);
    }
    // store value
    if (tid_in_vec == 0) {
      flat_atomic_add(y + reduce_row_id,
                      static_cast<Y_TYPE>(static_cast<COMPUTE_TYPE>(alpha) * static_cast<COMPUTE_TYPE>(sum)));
    }
  }

  reduce_row_id += vec_num;
  for (; reduce_row_id < reduce_end_row_id; reduce_row_id += vec_num) {
    const SHARE_MEM_INDEX_TYPE reduce_start_inx =
        max(static_cast<SHARE_MEM_INDEX_TYPE>(0),
            static_cast<SHARE_MEM_INDEX_TYPE>(row_offset[reduce_row_id] - bp_index * NNZ_PER_BLOCK));
    const SHARE_MEM_INDEX_TYPE reduce_end_inx =
        min(static_cast<SHARE_MEM_INDEX_TYPE>(NNZ_PER_BLOCK),
            static_cast<SHARE_MEM_INDEX_TYPE>(row_offset[reduce_row_id + 1] - bp_index * NNZ_PER_BLOCK));
    // reduce LDS via vectors.
    COMPUTE_TYPE sum = static_cast<COMPUTE_TYPE>(0);
    for (SHARE_MEM_INDEX_TYPE i = reduce_start_inx + tid_in_vec; i < reduce_end_inx; i += vec_size) {
      sum += _lds_shared_data[i];
    }
    // reduce in vector
    for (int i = vec_size >> 1; i > 0; i >>= 1) {
      sum += __shfl_down(sum, i, vec_size);
    }
    // store value
    if (tid_in_vec == 0) {
      flat_atomic_add(y + reduce_row_id,
                      static_cast<Y_TYPE>(static_cast<COMPUTE_TYPE>(alpha) * static_cast<COMPUTE_TYPE>(sum)));
    }
  }
}

template <typename BP_INDEX_TYPE, typename SHARE_MEM_DATA_TYPE, typename SHARE_MEM_INDEX_TYPE, typename NNZ_TYPE,
          typename ROW_TYPE, typename SCALE_DATA_TYPE, typename COMPUTE_TYPE, typename Y_TYPE, int NNZ_PER_BLOCK,
          int THREADS>
__device__ __forceinline__ void flat_reduce_direct(const int tid_in_block, const BP_INDEX_TYPE bp_index,
                                                   const ROW_TYPE reduce_start_row_id, const ROW_TYPE reduce_end_row_id,
                                                   const SCALE_DATA_TYPE alpha, const NNZ_TYPE *__restrict__ row_offset,
                                                   const SHARE_MEM_DATA_TYPE *__restrict__ _lds_shared_data,
                                                   Y_TYPE *__restrict__ y) {
  ROW_TYPE reduce_row_id = reduce_start_row_id + tid_in_block;
  if (reduce_row_id < reduce_end_row_id) {
    COMPUTE_TYPE sum = static_cast<COMPUTE_TYPE>(0);
    // what if it has a very long row? which means `reduce_start_row_id == reduce_end_row_id`.
    const SHARE_MEM_INDEX_TYPE reduce_start_inx =
        max(static_cast<SHARE_MEM_INDEX_TYPE>(0),
            static_cast<SHARE_MEM_INDEX_TYPE>(row_offset[reduce_row_id] - bp_index * NNZ_PER_BLOCK));
    const SHARE_MEM_INDEX_TYPE reduce_end_inx =
        min(static_cast<SHARE_MEM_INDEX_TYPE>(NNZ_PER_BLOCK),
            static_cast<SHARE_MEM_INDEX_TYPE>(row_offset[reduce_row_id + 1] - bp_index * NNZ_PER_BLOCK));
    for (SHARE_MEM_INDEX_TYPE i = reduce_start_inx; i < reduce_end_inx; i++) {
      sum += _lds_shared_data[i];
    }
    flat_atomic_add(y + reduce_row_id,
                    static_cast<Y_TYPE>(static_cast<COMPUTE_TYPE>(alpha) * static_cast<COMPUTE_TYPE>(sum)));
  }
  reduce_row_id += THREADS;
  for (; reduce_row_id < reduce_end_row_id; reduce_row_id += THREADS) {
    COMPUTE_TYPE sum = static_cast<COMPUTE_TYPE>(0);
    const SHARE_MEM_INDEX_TYPE reduce_start_inx =
        max(static_cast<SHARE_MEM_INDEX_TYPE>(0),
            static_cast<SHARE_MEM_INDEX_TYPE>(row_offset[reduce_row_id] - bp_index * NNZ_PER_BLOCK));
    const SHARE_MEM_INDEX_TYPE reduce_end_inx =
        min(static_cast<SHARE_MEM_INDEX_TYPE>(NNZ_PER_BLOCK),
            static_cast<SHARE_MEM_INDEX_TYPE>(row_offset[reduce_row_id + 1] - bp_index * NNZ_PER_BLOCK));
    for (SHARE_MEM_INDEX_TYPE i = reduce_start_inx; i < reduce_end_inx; i++) {
      sum += _lds_shared_data[i];
    }
    flat_atomic_add(y + reduce_row_id,
                    static_cast<Y_TYPE>(static_cast<COMPUTE_TYPE>(alpha) * static_cast<COMPUTE_TYPE>(sum)));
  }
}

// reduce results of a row in a vector with memory coalescing.
template <typename BP_INDEX_TYPE, typename SHARE_MEM_DATA_TYPE, typename SHARE_MEM_INDEX_TYPE, typename NNZ_TYPE,
          typename ROW_TYPE, typename SCALE_DATA_TYPE, typename COMPUTE_TYPE, typename Y_TYPE, int NNZ_PER_BLOCK,
          int THREADS, int VECTOR_SIZE>
__device__ __forceinline__ void flat_reduce_in_vector_with_mem_coalescing(
    const ROW_TYPE n_reduce_rows_num, const int tid_in_block, const BP_INDEX_TYPE bp_index,
    const ROW_TYPE reduce_start_row_id, const ROW_TYPE reduce_end_row_id, const SCALE_DATA_TYPE alpha,
    const NNZ_TYPE *__restrict__ row_offset, const SHARE_MEM_DATA_TYPE *__restrict__ _lds_shared_data,
    Y_TYPE *__restrict__ y) {
  // use `vec_num` vectors, each vector can process reduction of one row by involving `vec_size` threads.
  constexpr int vec_size = VECTOR_SIZE;
  constexpr int vec_num = THREADS / vec_size;
  const int vec_id = tid_in_block / vec_size;
  const int tid_in_vec = tid_in_block % vec_size;
  __shared__ COMPUTE_TYPE lds_y[THREADS / VECTOR_SIZE];

  ROW_TYPE reduce_row_id = reduce_start_row_id + vec_id;
  ROW_TYPE thread_reduce_row_id = reduce_start_row_id + tid_in_block;
  if (reduce_row_id < reduce_end_row_id) {
    const SHARE_MEM_INDEX_TYPE reduce_start_inx =
        max(static_cast<SHARE_MEM_INDEX_TYPE>(0),
            static_cast<SHARE_MEM_INDEX_TYPE>(row_offset[reduce_row_id] - bp_index * NNZ_PER_BLOCK));
    const SHARE_MEM_INDEX_TYPE reduce_end_inx =
        min(static_cast<SHARE_MEM_INDEX_TYPE>(NNZ_PER_BLOCK),
            static_cast<SHARE_MEM_INDEX_TYPE>(row_offset[reduce_row_id + 1] - bp_index * NNZ_PER_BLOCK));
    // reduce LDS via vectors.
    COMPUTE_TYPE sum = static_cast<COMPUTE_TYPE>(0);
    for (SHARE_MEM_INDEX_TYPE i = reduce_start_inx + tid_in_vec; i < reduce_end_inx; i += vec_size) {
      sum += static_cast<COMPUTE_TYPE>(_lds_shared_data[i]);
    }
    // reduce in vector
    for (int i = vec_size >> 1; i > 0; i >>= 1) {
      sum += __shfl_down(sum, i, vec_size);
    }
    // store sum value to y with memory coalescing
    if (tid_in_vec == 0) {
      lds_y[vec_id] = sum;
    }
  }
  __syncthreads();

  if (reduce_row_id < reduce_end_row_id) {
    // store sum value to y with memory coalescing
    if (tid_in_block < vec_num && thread_reduce_row_id < reduce_end_row_id) {
      const COMPUTE_TYPE local_sum = lds_y[tid_in_block];
      flat_atomic_add(y + thread_reduce_row_id,
                      static_cast<Y_TYPE>(static_cast<COMPUTE_TYPE>(alpha) * static_cast<COMPUTE_TYPE>(local_sum)));
    }
  }

  reduce_row_id += vec_num;
  thread_reduce_row_id += vec_num;
  const ROW_TYPE n_loop = (n_reduce_rows_num / vec_num) + (n_reduce_rows_num % vec_num == 0 ? 0 : 1) - 1;
  for (ROW_TYPE k = 0; k < n_loop; k++) {
    if (reduce_row_id < reduce_end_row_id) {
      const SHARE_MEM_INDEX_TYPE reduce_start_inx =
          max(static_cast<SHARE_MEM_INDEX_TYPE>(0),
              static_cast<SHARE_MEM_INDEX_TYPE>(row_offset[reduce_row_id] - bp_index * NNZ_PER_BLOCK));
      const SHARE_MEM_INDEX_TYPE reduce_end_inx =
          min(static_cast<SHARE_MEM_INDEX_TYPE>(NNZ_PER_BLOCK),
              static_cast<SHARE_MEM_INDEX_TYPE>(row_offset[reduce_row_id + 1] - bp_index * NNZ_PER_BLOCK));
      // reduce LDS via vectors.
      COMPUTE_TYPE sum = static_cast<COMPUTE_TYPE>(0);
      for (SHARE_MEM_INDEX_TYPE i = reduce_start_inx + tid_in_vec; i < reduce_end_inx; i += vec_size) {
        sum += static_cast<COMPUTE_TYPE>(_lds_shared_data[i]);
      }
      // reduce in vector
      for (int i = vec_size >> 1; i > 0; i >>= 1) {
        sum += __shfl_down(sum, i, vec_size);
      }

      if (tid_in_vec == 0) {
        lds_y[vec_id] = sum;
      }
    }

    __syncthreads();

    if (reduce_row_id < reduce_end_row_id) {
      // store sum value to y with memory coalescing
      if (thread_reduce_row_id < reduce_end_row_id && tid_in_block < vec_num) {
        const COMPUTE_TYPE local_sum = lds_y[tid_in_block];
        flat_atomic_add(y + thread_reduce_row_id,
                        static_cast<Y_TYPE>(static_cast<COMPUTE_TYPE>(alpha) * static_cast<COMPUTE_TYPE>(local_sum)));
      }
    }
    reduce_row_id += vec_num;
    thread_reduce_row_id += vec_num;
  }
}

template <typename BP_INDEX_TYPE, typename SHARE_MEM_DATA_TYPE, typename SHARE_MEM_INDEX_TYPE, typename NNZ_TYPE,
          typename ROW_TYPE, typename SCALE_DATA_TYPE, typename COMPUTE_TYPE, typename Y_TYPE, int NNZ_PER_BLOCK,
          int THREADS, int VECTOR_SIZE>
__device__ __forceinline__ void
flat_segment_reduce_in_vector(const ROW_TYPE n_reduce_rows_num, const int tid_in_block, const BP_INDEX_TYPE bp_index,
                              const ROW_TYPE reduce_start_row_id, const ROW_TYPE reduce_end_row_id,
                              const SCALE_DATA_TYPE alpha, const NNZ_TYPE *__restrict__ row_offset,
                              const SHARE_MEM_DATA_TYPE *__restrict__ _lds_shared_data, Y_TYPE *__restrict__ y,
                              KV<ROW_TYPE, Y_TYPE> *__restrict__ key_val) {
  // use `vec_num` vectors, each vector can process reduction of one row by involving `vec_size` threads.
  const int vec_size = VECTOR_SIZE;
  const int vec_num = THREADS / vec_size;
  const int vec_id = tid_in_block / vec_size;
  const int tid_in_vec = tid_in_block % vec_size;

  ROW_TYPE reduce_row_id = reduce_start_row_id + vec_id;
  COMPUTE_TYPE carry_out_part_sum;
  if (reduce_row_id < reduce_end_row_id) {
    const SHARE_MEM_INDEX_TYPE reduce_start_inx =
        max(static_cast<SHARE_MEM_INDEX_TYPE>(0),
            static_cast<SHARE_MEM_INDEX_TYPE>(row_offset[reduce_row_id] - bp_index * NNZ_PER_BLOCK));
    const SHARE_MEM_INDEX_TYPE reduce_end_inx =
        min(static_cast<SHARE_MEM_INDEX_TYPE>(NNZ_PER_BLOCK),
            static_cast<SHARE_MEM_INDEX_TYPE>(row_offset[reduce_row_id + 1] - bp_index * NNZ_PER_BLOCK));
    // reduce LDS via vectors.
    COMPUTE_TYPE sum = static_cast<COMPUTE_TYPE>(0);
    for (SHARE_MEM_INDEX_TYPE i = reduce_start_inx + tid_in_vec; i < reduce_end_inx; i += vec_size) {
      sum += _lds_shared_data[i];
    }
    // reduce in vector
    for (int i = vec_size >> 1; i > 0; i >>= 1) {
      sum += __shfl_down(sum, i, vec_size);
    }
    // store value
    if (tid_in_vec == 0) {
      // write result to y, but the last row in the block.
      if (reduce_row_id + 1 < reduce_end_row_id) {
        y[reduce_row_id] += static_cast<Y_TYPE>(static_cast<COMPUTE_TYPE>(alpha) * static_cast<COMPUTE_TYPE>(sum));
      }
      carry_out_part_sum = sum; // temp sum for the last row in current block.
    }
  }

  reduce_row_id += vec_num;
  for (; reduce_row_id < reduce_end_row_id; reduce_row_id += vec_num) {
    const SHARE_MEM_INDEX_TYPE reduce_start_inx =
        max(static_cast<SHARE_MEM_INDEX_TYPE>(0),
            static_cast<SHARE_MEM_INDEX_TYPE>(row_offset[reduce_row_id] - bp_index * NNZ_PER_BLOCK));
    const SHARE_MEM_INDEX_TYPE reduce_end_inx =
        min(static_cast<SHARE_MEM_INDEX_TYPE>(NNZ_PER_BLOCK),
            static_cast<SHARE_MEM_INDEX_TYPE>(row_offset[reduce_row_id + 1] - bp_index * NNZ_PER_BLOCK));
    // reduce LDS via vectors.
    COMPUTE_TYPE sum = static_cast<COMPUTE_TYPE>(0);
    for (SHARE_MEM_INDEX_TYPE i = reduce_start_inx + tid_in_vec; i < reduce_end_inx; i += vec_size) {
      sum += _lds_shared_data[i];
    }
    // reduce in vector
    for (int i = vec_size >> 1; i > 0; i >>= 1) {
      sum += __shfl_down(sum, i, vec_size);
    }
    // store value
    if (tid_in_vec == 0) {
      // write result to y, but the last row in the block.
      if (reduce_row_id + 1 < reduce_end_row_id) {
        y[reduce_row_id] += static_cast<Y_TYPE>(static_cast<COMPUTE_TYPE>(alpha) * static_cast<COMPUTE_TYPE>(sum));
      }
      carry_out_part_sum = sum; // temp sum for the last row in current block.
    }
  }

  // store the last row in current block to temp buffer
  if (tid_in_vec == 0) {
    // `reduce_row_id - vec_num` is the row just calculated by this vector
    if (reduce_row_id - vec_num + 1 == reduce_end_row_id) {
      key_val[blockIdx.x].key = reduce_row_id - vec_num;
      key_val[blockIdx.x].value =
          static_cast<Y_TYPE>(static_cast<COMPUTE_TYPE>(alpha) * static_cast<COMPUTE_TYPE>(carry_out_part_sum));
    }
  }
}

template <typename BP_INDEX_TYPE, typename SHARE_MEM_DATA_TYPE, typename SHARE_MEM_INDEX_TYPE, typename NNZ_TYPE,
          typename ROW_TYPE, typename SCALE_DATA_TYPE, typename COMPUTE_TYPE, typename Y_TYPE, int NNZ_PER_BLOCK,
          int THREADS>
__device__ __forceinline__ void
flat_segment_reduce_direct(const int tid_in_block, const BP_INDEX_TYPE bp_index, const ROW_TYPE reduce_start_row_id,
                           const ROW_TYPE reduce_end_row_id, const SCALE_DATA_TYPE alpha,
                           const NNZ_TYPE *__restrict__ row_offset,
                           const SHARE_MEM_DATA_TYPE *__restrict__ _lds_shared_data, Y_TYPE *__restrict__ y,
                           KV<ROW_TYPE, Y_TYPE> *__restrict__ key_val) {
  ROW_TYPE reduce_row_id = reduce_start_row_id + tid_in_block;
  COMPUTE_TYPE carry_out_part_sum;
  if (reduce_row_id < reduce_end_row_id) {
    COMPUTE_TYPE sum = static_cast<COMPUTE_TYPE>(0);
    // what if it has a very long row? which means `reduce_start_row_id == reduce_end_row_id`.
    const SHARE_MEM_INDEX_TYPE reduce_start_inx =
        max(static_cast<SHARE_MEM_INDEX_TYPE>(0),
            static_cast<SHARE_MEM_INDEX_TYPE>(row_offset[reduce_row_id] - bp_index * NNZ_PER_BLOCK));
    const SHARE_MEM_INDEX_TYPE reduce_end_inx =
        min(static_cast<SHARE_MEM_INDEX_TYPE>(NNZ_PER_BLOCK),
            static_cast<SHARE_MEM_INDEX_TYPE>(row_offset[reduce_row_id + 1] - bp_index * NNZ_PER_BLOCK));
    for (SHARE_MEM_INDEX_TYPE i = reduce_start_inx; i < reduce_end_inx; i++) {
      sum += _lds_shared_data[i];
    }

    if (reduce_row_id + 1 < reduce_end_row_id) {
      y[reduce_row_id] += static_cast<Y_TYPE>(static_cast<COMPUTE_TYPE>(alpha) * static_cast<COMPUTE_TYPE>(sum));
    }
    carry_out_part_sum = sum;
  }

  reduce_row_id += THREADS;
  for (; reduce_row_id < reduce_end_row_id; reduce_row_id += THREADS) {
    COMPUTE_TYPE sum = static_cast<COMPUTE_TYPE>(0);
    const SHARE_MEM_INDEX_TYPE reduce_start_inx =
        max(static_cast<SHARE_MEM_INDEX_TYPE>(0),
            static_cast<SHARE_MEM_INDEX_TYPE>(row_offset[reduce_row_id] - bp_index * NNZ_PER_BLOCK));
    const SHARE_MEM_INDEX_TYPE reduce_end_inx =
        min(static_cast<SHARE_MEM_INDEX_TYPE>(NNZ_PER_BLOCK),
            static_cast<SHARE_MEM_INDEX_TYPE>(row_offset[reduce_row_id + 1] - bp_index * NNZ_PER_BLOCK));
    for (SHARE_MEM_INDEX_TYPE i = reduce_start_inx; i < reduce_end_inx; i++) {
      sum += _lds_shared_data[i];
    }

    if (reduce_row_id + 1 < reduce_end_row_id) {
      y[reduce_row_id] += static_cast<Y_TYPE>(static_cast<COMPUTE_TYPE>(alpha) * static_cast<COMPUTE_TYPE>(sum));
    }
    carry_out_part_sum = sum;
  }

  // `reduce_row_id - THREADS` is the row just calculated by this vector
  if (reduce_row_id - THREADS + 1 == reduce_end_row_id) {
    key_val[blockIdx.x].key = reduce_row_id - THREADS;
    key_val[blockIdx.x].value =
        static_cast<Y_TYPE>(static_cast<COMPUTE_TYPE>(alpha) * static_cast<COMPUTE_TYPE>(carry_out_part_sum));
  }
}

template <typename ROW, typename Y_TYPE, typename HIP_BLOCK_NUM_TYPE>
__global__ void flat_reduce_carry_out_kernel(Y_TYPE *__restrict__ y, KV<ROW, Y_TYPE> *__restrict__ key_val,
                                             const HIP_BLOCK_NUM_TYPE HIP_BLOCKS) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) {
    // a very simple kernel, only one thread works.
    for (int i = 0; i < HIP_BLOCKS; i++) {
      y[key_val[i].key] += key_val[i].value;
    }
  }
}

// Segmented block reduction kernel
template <uint32_t BLOCKSIZE, typename I, typename T>
__device__ __forceinline__ void segmented_blockreduce(const I *rows, T *vals) {
  const int tid = hipThreadIdx_x;

#pragma unroll
  for (uint32_t j = 1; j < BLOCKSIZE; j <<= 1) {
    T val = static_cast<T>(0);
    if (tid >= j) {
      if (rows[tid] == rows[tid - j]) {
        val = vals[tid - j];
      }
    }
    __syncthreads();

    vals[tid] = vals[tid] + val;
    __syncthreads();
  }
  return;
}

// Do the final block reduction of the block reduction buffers back into global memory
template <uint32_t BLOCKSIZE, typename I, typename Y, typename T>
__global__ __launch_bounds__(512) void final_segmented_loops_reduce_device(I nblocks,
                                                                           const KV<I, T> *__restrict__ key_val,
                                                                           Y *__restrict__ y) {
  const int tid = hipThreadIdx_x;

  // Shared memory to hold row indices and values for segmented reduction
  __shared__ I shared_row[BLOCKSIZE];
  __shared__ T shared_val[BLOCKSIZE];

  // Loop over blocks that are subject for segmented reduction
  for (I i = 0; i < nblocks; i += BLOCKSIZE) {
    // Copy data to reduction buffers
    shared_row[tid] = (tid + i < nblocks) ? key_val[tid + i].key : -1;
    shared_val[tid] = (tid + i < nblocks) ? key_val[tid + i].value : static_cast<T>(0);

    __syncthreads();

    // Do segmented block reduction
    segmented_blockreduce<BLOCKSIZE>(shared_row, shared_val);

    // Add reduced sum to y if valid
    const I row = shared_row[tid];
    const I rowp1 = (tid < BLOCKSIZE - 1) ? shared_row[tid + 1] : -1;

    if (row != rowp1 && row >= 0) {
      y[row] = y[row] + shared_val[tid];
    }

    __syncthreads();
  }
  return;
}

template <uint32_t BLOCKSIZE, typename I, typename Y, typename T>
__global__ __launch_bounds__(512) void segmented_loops_reduce_device(I nblocks, const KV<I, T> *__restrict__ key_val,
                                                                     Y *__restrict__ y,
                                                                     KV<I, T> *__restrict__ temp_data) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid_in_block = threadIdx.x;

  // Shared memory to hold row indices and values for segmented reduction
  __shared__ I shared_row[BLOCKSIZE];
  __shared__ T shared_val[BLOCKSIZE];

  // Loop over blocks that are subject for segmented reduction
  if (true) {
    constexpr int i = 0;
    // Copy data to reduction buffers
    shared_row[tid_in_block] = (tid + i < nblocks) ? key_val[tid + i].key : -1;
    shared_val[tid_in_block] = (tid + i < nblocks) ? key_val[tid + i].value : static_cast<T>(0);

    __syncthreads();

    // Do segmented block reduction
    segmented_blockreduce<BLOCKSIZE>(shared_row, shared_val);

    // Add reduced sum to y if valid
    const I row = shared_row[tid_in_block];

    if (tid_in_block == BLOCKSIZE - 1) {
      // the last thread in the block writes to temp buffer
      if (row != -1) {
        temp_data[blockIdx.x].key = row;
        temp_data[blockIdx.x].value = shared_val[tid_in_block];
      }
    } else {
      const I rowp1 = (tid_in_block < BLOCKSIZE - 1) ? shared_row[tid_in_block + 1] : -1;
      if (row != rowp1 && row >= 0) {
        y[row] = y[row] + shared_val[tid_in_block];
      }
    }
    __syncthreads();
  }
  return;
}

template <typename ROW, typename Y_TYPE, typename HIP_BLOCK_NUM_TYPE, int THREADS_PER_BLOCK>
inline void flat_reduce_carry_out(hipStream_t stream, Y_TYPE *__restrict__ y, KV<ROW, Y_TYPE> *__restrict__ key_val,
                                  const HIP_BLOCK_NUM_TYPE HIP_BLOCKS) {
  if (HIP_BLOCKS < 2 * THREADS_PER_BLOCK) {
    final_segmented_loops_reduce_device<THREADS_PER_BLOCK, ROW, Y_TYPE, Y_TYPE>
        <<<1, THREADS_PER_BLOCK, 0, stream>>>(HIP_BLOCKS, key_val, y);
  } else {
    int NEW_HIP_BLOCKS = (HIP_BLOCKS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    KV<ROW, Y_TYPE> *temp_data = key_val + HIP_BLOCKS;
    segmented_loops_reduce_device<THREADS_PER_BLOCK, ROW, Y_TYPE, Y_TYPE>
        <<<NEW_HIP_BLOCKS, THREADS_PER_BLOCK, 0, stream>>>(HIP_BLOCKS, key_val, y, temp_data);
    hipStreamSynchronize(stream);
    flat_reduce_carry_out<ROW, Y_TYPE, HIP_BLOCK_NUM_TYPE, THREADS_PER_BLOCK>(stream, y, temp_data, NEW_HIP_BLOCKS);
  }
  return;
}

#endif // USPARSE_HIP_USPARSE_FLAT_KERNELS_IMP_REDUCE_HPP
