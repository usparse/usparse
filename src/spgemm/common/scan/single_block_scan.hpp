#ifndef COMMON_SINGLE_BLOCK_SCAN_HPP
#define COMMON_SINGLE_BLOCK_SCAN_HPP

#include "../check/hip_check.hpp"
#include "scan_in_device.hpp"
#include <hip/hip_runtime.h>

namespace single_block_scan {
constexpr size_t BLOCKS = 1;
constexpr size_t THREADS = 1024;
constexpr size_t VEC_SIZE = __WF_SIZE__;

template <typename T, size_t THREADS, size_t VEC_SIZE>
__global__ void __launch_bounds__(THREADS)
    single_block_scan_kernel(const size_t input_size, const T *input, T *output) {
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
  if (blockIdx.x == 0) {
    if (block_thread_id == 0) {
      shared_base_sum = static_cast<T>(0);
    }
    __syncthreads();
    for (int i = block_thread_id; i < input_size; i += THREADS) {
      T local_sum = input[i];
      scan_in_vector_device<T, VEC_SIZE>(&local_sum);
      if (vector_thread_id == VEC_SIZE - 1) {
        *lds_vec_sum = local_sum;
      }
      __syncthreads();
      // scan for lds
      if (block_vector_id == 0) {
        T lds_scan_sum = static_cast<T>(0);
        for (int j = vector_thread_id; j < THREADS / VEC_SIZE; j += VEC_SIZE) {
          T vec_local_sum = shared_vector_reduce[j];
          scan_in_vector_device<T, VEC_SIZE>(&vec_local_sum);
          vec_local_sum += lds_scan_sum;
          shared_vector_reduce[j] = vec_local_sum;
          lds_scan_sum = __shfl(vec_local_sum, VEC_SIZE - 1, VEC_SIZE);
        }
      }
      __syncthreads();
      // add sum in lds
      output[i] = shared_base_sum + local_sum +
                  (block_vector_id > 0 ? shared_vector_reduce[block_vector_id - 1] : static_cast<T>(0));
      __syncthreads();
      // update base sum
      if (block_thread_id == 0) {
        shared_base_sum += shared_vector_reduce[THREADS / VEC_SIZE - 1];
      }
      __syncthreads();
    }
  }
}

template <typename T> size_t single_block_scan_buffer_size(const size_t input_size) { return 0; }

template <typename T> void single_block_scan_impl(const size_t input_size, const T *input, T *output) {
  (single_block_scan_kernel<T, THREADS, VEC_SIZE>)<<<BLOCKS, THREADS>>>(input_size, input, output);
}

template <typename T> void single_block_scan_wrapper(const size_t input_size, const T *input, T *output) {
  (single_block_scan_kernel<T, THREADS, VEC_SIZE>)<<<BLOCKS, THREADS>>>(input_size, input, output);
}
} // namespace single_block_scan

#endif // COMMON_SINGLE_BLOCK_SCAN_HPP