#ifndef SPGEMM_HIP_KOKKOS_SPLIT_GROUP_KERNEL_HPP
#define SPGEMM_HIP_KOKKOS_SPLIT_GROUP_KERNEL_HPP

#include "../common/scan/scan_in_device.hpp"
#include "split_config.h"
#include <hip/hip_runtime.h>

// since C++17
template <typename I, size_t GROUP_NUM, const I (&GROUP_THRESHOLD)[GROUP_NUM], size_t N, size_t THREADS>
__global__ void __launch_bounds__(THREADS)
    split_group_part1_kernel(const size_t m, const I *row_ptr, int *temp_buffer) {
  // block id in global
  const int global_block_id = blockIdx.x;
  // block num in global
  const int global_block_num = gridDim.x;
  // thread id in block
  const int block_thread_id = threadIdx.x;
  // thread num in block
  constexpr int block_thread_num = THREADS;
  // wf id in block
  const int block_wf_id = block_thread_id / __WF_SIZE__;
  // thread id in wf
  const int wf_thread_id = block_thread_id % __WF_SIZE__;
  // thread num in wf
  constexpr int wf_thread_num = __WF_SIZE__;

  constexpr size_t ROWS_PER_BLOCK = THREADS * N;
  const int block_row_begin = ROWS_PER_BLOCK * global_block_id;
  const int block_row_end = min(block_row_begin + ROWS_PER_BLOCK, m);

  __shared__ int block_split_count[GROUP_NUM];
  int local_split_count[GROUP_NUM];

  for (int i = block_thread_id; i < GROUP_NUM; i += block_thread_num) {
    block_split_count[i] = 0;
  }

  for (int i = 0; i < GROUP_NUM; ++i) {
    local_split_count[i] = 0;
  }

  for (int row = block_row_begin + block_thread_id; row < block_row_end; row += block_thread_num) {
    I row_nnz = row_ptr[row + 1] - row_ptr[row];
    if (row_nnz > 0) {
#pragma unroll GROUP_NUM
      for (int i = 0; i < GROUP_NUM; ++i) {
        if (row_nnz <= GROUP_THRESHOLD[i]) {
          local_split_count[i]++;
          break;
        }
      }
    }
  }
  __syncthreads();

// reduce
#pragma unroll GROUP_NUM
  for (int i = 0; i < GROUP_NUM; ++i) {
    for (int j = __WF_SIZE__ >> 1; j > 0; j >>= 1) {
      local_split_count[i] += __shfl_down(local_split_count[i], j, __WF_SIZE__);
    }
    if (wf_thread_id == 0) {
      atomicAdd(&block_split_count[i], local_split_count[i]);
    }
  }
  __syncthreads();
  for (int i = block_thread_id; i < GROUP_NUM; i += block_thread_num) {
    temp_buffer[global_block_num * i + global_block_id] = block_split_count[i];
  }
}

template <typename T, size_t VEC_SIZE, size_t THREADS>
__global__ void __launch_bounds__(THREADS)
    split_group_part2_kernel(const size_t input_size, const size_t step_width, T *scan_ptr, T *sum_ptr) {
  // thread id in global
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  // thread id in block
  const int block_thread_id = threadIdx.x;
  // vector id in block
  const int block_vector_id = block_thread_id / VEC_SIZE;
  // thread id in vector
  const int vector_thread_id = block_thread_id % VEC_SIZE;

  __shared__ T shared_vector_reduce[THREADS / VEC_SIZE];
  __shared__ T shared_base_sum;
  T *lds_vec_sum = shared_vector_reduce + block_vector_id;

  const T *input = scan_ptr + step_width * blockIdx.x;
  T *output = scan_ptr + step_width * blockIdx.x;
  T *block_sum = sum_ptr + blockIdx.x;
  if (block_thread_id == 0) {
    shared_base_sum = static_cast<T>(0);
  }
  __syncthreads();
  for (int i = block_thread_id; i < input_size; i += THREADS) {
    T local_sum = input[i];
    common::scan_in_vector_device<T, VEC_SIZE>(&local_sum);
    if (vector_thread_id == VEC_SIZE - 1) {
      *lds_vec_sum = local_sum;
    }
    __syncthreads();
    // scan for lds
    if (block_vector_id == 0) {
      T lds_scan_sum = static_cast<T>(0);
      for (int j = vector_thread_id; j < THREADS / VEC_SIZE; j += VEC_SIZE) {
        T vec_local_sum = shared_vector_reduce[j];
        common::scan_in_vector_device<T, VEC_SIZE>(&vec_local_sum);
        vec_local_sum += lds_scan_sum;
        shared_vector_reduce[j] = vec_local_sum;
        lds_scan_sum = __shfl(vec_local_sum, VEC_SIZE - 1, VEC_SIZE);
      }
    }
    __syncthreads();
    // add sum in lds
    T tmp = shared_base_sum + local_sum +
            (block_vector_id > 0 ? shared_vector_reduce[block_vector_id - 1] : static_cast<T>(0));
    output[i] = tmp;
    if (i == input_size - 1) {
      *block_sum = tmp;
    }
    __syncthreads();
    // update base sum
    if (block_thread_id == 0) {
      shared_base_sum += shared_vector_reduce[THREADS / VEC_SIZE - 1];
    }
    __syncthreads();
  }
}

template <typename I, size_t GROUP_NUM, const I (&GROUP_THRESHOLD)[GROUP_NUM], size_t N, size_t THREADS>
__global__ void __launch_bounds__(THREADS)
    split_group_part3_kernel(const size_t m, const I *row_ptr, I *group_row_ptr, int *temp_buffer) {
  // block id in global
  const int global_block_id = blockIdx.x;
  // block num in global
  const int global_block_num = gridDim.x;
  // thread id in block
  const int block_thread_id = threadIdx.x;
  // thread num in block
  constexpr int block_thread_num = THREADS;
  // wf id in block
  const int block_wf_id = block_thread_id / __WF_SIZE__;
  // thread id in wf
  const int wf_thread_id = block_thread_id % __WF_SIZE__;
  // thread num in wf
  constexpr int wf_thread_num = __WF_SIZE__;

  constexpr size_t ROWS_PER_BLOCK = THREADS * N;

  __shared__ I block_group_row[THREADS * GROUP_NUM];
  __shared__ int block_insert_idx[GROUP_NUM];
  __shared__ int block_split_count[GROUP_NUM];

  // init
  for (int i = block_thread_id; i < GROUP_NUM; i += block_thread_num) {
    if (global_block_id > 0) {
      block_split_count[i] = temp_buffer[global_block_num * i + global_block_id - 1];
    } else {
      block_split_count[i] = 0;
    }
  }

  for (int r = 0; r < N; r++) {
    const int block_round_begin = ROWS_PER_BLOCK * global_block_id + r * THREADS;
    const int block_round_end = min(block_round_begin + THREADS, m);

    // init
    for (int i = block_thread_id; i < GROUP_NUM; i += block_thread_num) {
      block_insert_idx[i] = 0;
    }
    __syncthreads();
    for (int row = block_round_begin + block_thread_id; row < block_round_end; row += block_thread_num) {
      I row_nnz = row_ptr[row + 1] - row_ptr[row];
      if (row_nnz > 0) {
#pragma unroll GROUP_NUM
        for (int i = 0; i < GROUP_NUM; ++i) {
          if (row_nnz <= GROUP_THRESHOLD[i]) {
            const int idx = atomicAdd(&block_insert_idx[i], 1);
            block_group_row[THREADS * i + idx] = row;
            break;
          }
        }
      }
    }
    __syncthreads();
    // write
    for (int i = 0; i < GROUP_NUM; ++i) {
      for (int idx = block_thread_id; idx < block_insert_idx[i]; idx += block_thread_num) {
        group_row_ptr[m * i + block_split_count[i] + idx] = block_group_row[i * THREADS + idx];
      }
    }
    __syncthreads();
    // reduce
    for (int i = block_thread_id; i < GROUP_NUM; i += block_thread_num) {
      block_split_count[i] += block_insert_idx[i];
    }
    __syncthreads();
  }
}

#endif // SPGEMM_HIP_KOKKOS_SPLIT_GROUP_KERNEL_HPP
