#ifndef REDUCE_THEN_SCAN_HPP
#define REDUCE_THEN_SCAN_HPP

#include "../check/hip_check.hpp"
#include "scan_in_device.hpp"
#include "single_block_scan.hpp"
#include <hip/hip_runtime.h>

namespace reduce_then_scan {
constexpr size_t THREADS = 1024;
constexpr size_t VEC_SIZE = 64;

template <typename T, size_t THREADS>
__global__ void __launch_bounds__(THREADS)
    reduce_then_scan_part1(const size_t input_size, const T *input, T *temp_buffer) {
  // thread id in global
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  // block id in global
  const int global_block_id = blockIdx.x;
  // block num in global
  const int global_block_num = gridDim.x;
  // thread id in block
  const int block_thread_id = threadIdx.x;
  // thread id in wf
  const int wf_thread_id = block_thread_id % __WF_SIZE__;

  const int idx_per_block = input_size / global_block_num + (input_size % global_block_num == 0 ? 0 : 1);
  const int block_idx_begin = blockIdx.x * idx_per_block;
  const int block_idx_end = min(input_size, block_idx_begin + idx_per_block);

  __shared__ T shared_block_sum;
  if (block_thread_id == 0) {
    shared_block_sum = static_cast<T>(0);
  }
  T local_sum = static_cast<T>(0);
  for (int idx = block_idx_begin + block_thread_id; idx < block_idx_end; idx += THREADS) {
    local_sum += input[idx];
  }
  for (int i = __WF_SIZE__ >> 1; i > 0; i >>= 1) {
    local_sum += __shfl_down(local_sum, i, __WF_SIZE__);
  }
  if (wf_thread_id == 0) {
    atomicAdd(&shared_block_sum, local_sum);
  }
  __syncthreads();
  if (block_thread_id == 0) {
    temp_buffer[global_block_id] = shared_block_sum;
  }
}

template <typename T, size_t THREADS, size_t VEC_SIZE>
__global__ void __launch_bounds__(THREADS)
    reduce_then_scan_part2(const size_t input_size, const T *input, T *output, const T *temp_buffer) {
  // thread id in global
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  // block id in global
  const int global_block_id = blockIdx.x;
  // block num in global
  const int global_block_num = gridDim.x;
  // thread id in block
  const int block_thread_id = threadIdx.x;
  // vector id in block
  const int block_vector_id = block_thread_id / VEC_SIZE;
  // thread id in vector
  const int vector_thread_id = block_thread_id % VEC_SIZE;

  const int idx_per_block = input_size / global_block_num + (input_size % global_block_num == 0 ? 0 : 1);
  const int block_idx_begin = blockIdx.x * idx_per_block;
  const int block_idx_end = min(input_size, block_idx_begin + idx_per_block);

  __shared__ T shared_vector_reduce[THREADS / VEC_SIZE];
  __shared__ T shared_base_sum;
  T *lds_vec_sum = shared_vector_reduce + block_vector_id;

  if (block_thread_id == 0) {
    shared_base_sum = static_cast<T>(0);
  }
  __syncthreads();
  const int block_rounds =
      (block_idx_end - block_idx_begin) / THREADS + ((block_idx_end - block_idx_begin) % THREADS == 0 ? 0 : 1);
  for (int round = 0; round < block_rounds; ++round) {
    const int idx = block_idx_begin + THREADS * round + block_thread_id;
    T local_sum = idx < block_idx_end ? input[idx] : static_cast<T>(0);
    scan_in_vector_device<T, VEC_SIZE, __WF_SIZE__>(&local_sum);
    if (vector_thread_id == VEC_SIZE - 1) {
      *lds_vec_sum = local_sum;
    }
    __syncthreads();
    // scan for lds
    if (block_vector_id == 0) {
      T lds_scan_sum = static_cast<T>(0);
      for (int j = vector_thread_id; j < THREADS / VEC_SIZE; j += VEC_SIZE) {
        T vec_local_sum = shared_vector_reduce[j];
        scan_in_vector_device<T, VEC_SIZE, __WF_SIZE__>(&vec_local_sum);
        vec_local_sum += lds_scan_sum;
        shared_vector_reduce[j] = vec_local_sum;
        lds_scan_sum = __shfl(vec_local_sum, VEC_SIZE - 1, VEC_SIZE);
      }
    }
    __syncthreads();
    // add sum in lds
    if (idx < block_idx_end) {
      T value = shared_base_sum + local_sum;
      value += block_vector_id > 0 ? shared_vector_reduce[block_vector_id - 1] : static_cast<T>(0);
      value += global_block_id > 0 ? temp_buffer[global_block_id - 1] : static_cast<T>(0);
      output[idx] = value;
    }
    __syncthreads();
    // update base sum
    if (block_thread_id == 0) {
      shared_base_sum += shared_vector_reduce[THREADS / VEC_SIZE - 1];
    }
    __syncthreads();
  }
}

template <typename T> size_t reduce_then_scan_buffer_size(const size_t input_size) {

  const size_t BLOCKS = input_size / THREADS + (input_size % THREADS == 0 ? 0 : 1);
  const size_t BUFFER_SIZE = BLOCKS * sizeof(T) * 3;
  return BUFFER_SIZE;
}

template <typename T> void reduce_then_scan_impl(const size_t input_size, const T *input, T *output, T *temp_buffer) {
  if (input_size < THREADS) {
    (single_block_scan::single_block_scan_kernel<T, THREADS, VEC_SIZE>)<<<1, THREADS>>>(input_size, input, output);
  } else {
    const size_t BLOCKS = input_size / THREADS + (input_size % THREADS == 0 ? 0 : 1);

    (reduce_then_scan_part1<T, THREADS>)<<<BLOCKS, THREADS>>>(input_size, input, temp_buffer);
    reduce_then_scan_impl<T>(BLOCKS, temp_buffer, temp_buffer + BLOCKS, temp_buffer + BLOCKS * 2);
    (reduce_then_scan_part2<T, THREADS, VEC_SIZE>)<<<BLOCKS, THREADS>>>(input_size, input, output,
                                                                        temp_buffer + BLOCKS);
  }
}

template <typename T> void reduce_then_scan_wrapper(const size_t input_size, const T *input, T *output) {
  T *temp_buffer = nullptr;
  const size_t temp_buffer_size = reduce_then_scan_buffer_size<int>(input_size);
  HIP_CHECK(hipMalloc((void **)(&temp_buffer), temp_buffer_size))
  reduce_then_scan_impl<T>(input_size, input, output, temp_buffer);
  HIP_CHECK(hipFree(temp_buffer))
}
} // namespace reduce_then_scan

#endif // REDUCE_THEN_SCAN_HPP