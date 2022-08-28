#ifndef SPGEMM_KOKKOS_MAX_NUM_KERNEL_HPP
#define SPGEMM_KOKKOS_MAX_NUM_KERNEL_HPP

#include <hip/hip_runtime.h>

template <typename T, size_t VEC_SIZE, size_t R, size_t THREADS>
__global__ void __launch_bounds__(THREADS) max_num_vector_kernel(const size_t len, const T *array, T *buffer) {
  // thread id in global
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  // thread num in global
  const int global_thread_num = blockDim.x * gridDim.x;
  // block id in global
  const int global_block_id = blockIdx.x;
  // block num in global
  const int global_block_num = gridDim.x;
  // thread id in block
  const int block_thread_id = threadIdx.x;
  // thread num in block
  constexpr int block_thread_num = THREADS;
  // vector id in block
  const int block_vector_id = block_thread_id / VEC_SIZE;
  // vector num in block
  constexpr int block_vector_num = THREADS / VEC_SIZE;
  // thread id in vector
  const int vector_thread_id = block_thread_id % VEC_SIZE;
  // thread num in vector
  constexpr int vector_thread_num = VEC_SIZE;

  __shared__ T shared_block_max[block_vector_num];
  const int process_each_block = THREADS * R;
  const int block_max_begin = global_block_id * process_each_block;
  const int block_max_end = min(block_max_begin + process_each_block, len);

  T max_num = 0;
#pragma unroll
  for (int i = 0; i < R; ++i) {
    const int vector_max_begin = block_max_begin + block_vector_id * VEC_SIZE + THREADS * i;
    const int arr_idx = vector_max_begin + vector_thread_id;
    T tmp = arr_idx < block_max_end ? array[arr_idx] : 0;
    max_num = max(max_num, tmp);
    // wavefront reduce
#pragma unroll
    for (int j = 1; j < VEC_SIZE; j <<= 1) {
      max_num = max(max_num, __shfl_up(max_num, j, VEC_SIZE));
    }
    if (vector_thread_id == VEC_SIZE - 1) {
      shared_block_max[block_vector_id] = max_num;
    }
  }
  __syncthreads();
  // block_reduce
  if (block_vector_id == 0) {
    max_num = 0;
    constexpr int rounds = block_vector_num / VEC_SIZE + ((block_vector_num) % VEC_SIZE == 0 ? 0 : 1);
#pragma unroll
    for (int i = 0; i < rounds; ++i) {
      const int idx = i * VEC_SIZE + vector_thread_id;
      // wavefront reduce
      T tmp = idx < block_vector_num ? shared_block_max[idx] : 0;
      max_num = max(max_num, tmp);
#pragma unroll
      for (int j = 1; j < VEC_SIZE; j <<= 1) {
        max_num = max(max_num, __shfl_up(max_num, j, VEC_SIZE));
      }
    }
    // write back
    if (vector_thread_id == VEC_SIZE - 1) {
      buffer[global_block_id] = max_num;
    }
  }
}

template <typename T, size_t VEC_SIZE, size_t THREADS>
__global__ void __launch_bounds__(THREADS) max_num_once_wf_kernel(const size_t len, const T *buffer, T *max_num) {
  // thread id in global
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  // thread num in global
  const int global_thread_num = blockDim.x * gridDim.x;
  // block id in global
  const int global_block_id = blockIdx.x;
  // block num in global
  const int global_block_num = gridDim.x;
  // thread id in block
  const int block_thread_id = threadIdx.x;
  // thread num in block
  constexpr int block_thread_num = THREADS;
  // vector id in block
  const int block_vector_id = block_thread_id / VEC_SIZE;
  // vector num in block
  constexpr int block_vector_num = THREADS / VEC_SIZE;
  // thread id in vector
  const int vector_thread_id = block_thread_id % VEC_SIZE;
  // thread num in vector
  constexpr int vector_thread_num = VEC_SIZE;

  if (block_vector_id == 0) {
    T local_max_num = 0;
    const int rounds = len / VEC_SIZE + (len % VEC_SIZE == 0 ? 0 : 1);
    for (int i = 0; i < rounds; ++i) {
      const int idx = i * VEC_SIZE + vector_thread_id;
      // wavefront reduce
      T tmp = idx < len ? buffer[idx] : 0;
      local_max_num = max(local_max_num, tmp);
#pragma unroll
      for (int j = 1; j < VEC_SIZE; j <<= 1) {
        local_max_num = max(local_max_num, __shfl_up(local_max_num, j, VEC_SIZE));
      }
    }
    if (vector_thread_id == VEC_SIZE - 1) {
      *max_num = local_max_num;
    }
  }
}

template <typename T> __global__ void max_num_single_thread_kernel(const size_t len, const T *array, T *max_num) {
  // thread id in global
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  if (global_thread_id == 0) {
    T local_max_num = 0;
    for (int I = 0; I < len; ++I) {
      local_max_num = max(local_max_num, array[I]);
    }
    *max_num = local_max_num;
  }
}

#endif // SPGEMM_KOKKOS_MAX_NUM_KERNEL_HPP
