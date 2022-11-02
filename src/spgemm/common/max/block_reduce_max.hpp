#ifndef BLOCK_REDUCE_MAX_KERNEL_HPP
#define BLOCK_REDUCE_MAX_KERNEL_HPP

#include "../check/hip_check.hpp"
#include "../tools/set_val.hpp"
#include <hip/hip_runtime.h>
#include <limits>

namespace common {

template <typename T> size_t block_reduce_max_buffer_size(const size_t input_size) {
  constexpr size_t THREADS = 256;
  constexpr size_t ITEMS_PER_THREAD = 4;
  constexpr size_t ITEMS_PER_BLOCK = ITEMS_PER_THREAD * THREADS;
  const size_t BLOCKS = input_size / ITEMS_PER_BLOCK + (input_size % ITEMS_PER_BLOCK == 0 ? 0 : 1);
  return BLOCKS * sizeof(T);
}

template <typename T, size_t THREADS, size_t ITEMS_PER_THREAD>
__global__ void __launch_bounds__(THREADS)
    block_reduce_max_kernel(const size_t n, const T *input, T *temp_buffer, T *output) {
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
  // wf num in block
  constexpr int block_wf_num = THREADS / __WF_SIZE__;
  // thread id in wf
  const int wf_thread_id = block_thread_id % __WF_SIZE__;

  constexpr size_t ITEMS_PER_BLOCK = THREADS * ITEMS_PER_THREAD;

  const int block_idx_begin = global_block_id * ITEMS_PER_BLOCK;
  const int block_idx_end = min(block_idx_begin + ITEMS_PER_BLOCK, n);

  __shared__ T lds_block_reduce_num[block_wf_num];

  T max_num = std::numeric_limits<T>::min();

  for (int idx = block_idx_begin + block_thread_id; idx < block_idx_end; idx += block_thread_num) {
    max_num = max(input[idx], max_num);
  }
  for (int i = __WF_SIZE__ >> 1; i > 0; i >>= 1) {
    max_num = max(__shfl_down(max_num, i, __WF_SIZE__), max_num);
  }
  if (wf_thread_id == 0) {
    lds_block_reduce_num[block_wf_id] = max_num;
  }
  __syncthreads();

  if (block_wf_id == 0) {
    for (int idx = wf_thread_id; idx < THREADS / __WF_SIZE__; idx += __WF_SIZE__) {
      max_num = max(lds_block_reduce_num[idx], max_num);
    }
    for (int i = __WF_SIZE__ >> 1; i > 0; i >>= 1) {
      max_num = max(__shfl_down(max_num, i, __WF_SIZE__), max_num);
    }
    int flat_block_id = 0;
    if (wf_thread_id == 0) {
      int *counter = reinterpret_cast<int *>(output);
      temp_buffer[global_block_id] = max_num;
      __threadfence();
      flat_block_id = atomicAdd(counter, 1);
    }
    flat_block_id = __shfl(flat_block_id, 0, __WF_SIZE__);
    if (flat_block_id == global_block_num - 1) {
      for (int idx = wf_thread_id; idx < global_block_num; idx += __WF_SIZE__) {
        max_num = max(temp_buffer[idx], max_num);
      }
      for (int i = __WF_SIZE__ >> 1; i > 0; i >>= 1) {
        max_num = max(__shfl_down(max_num, i, __WF_SIZE__), max_num);
      }
      if (wf_thread_id == 0) {
        *output = max_num;
      }
    }
  }
}

template <typename T>
void block_reduce_max_impl(const size_t input_size, const T *input, T *output, void *temp_buffer) {
  static_assert(sizeof(T) >= sizeof(int), "items must more than 32bits");

  constexpr size_t THREADS = 256;
  constexpr size_t ITEMS_PER_THREAD = 4;
  constexpr size_t ITEMS_PER_BLOCK = ITEMS_PER_THREAD * THREADS;
  const size_t BLOCKS = input_size / ITEMS_PER_BLOCK + (input_size % ITEMS_PER_BLOCK == 0 ? 0 : 1);

  T *block_buffer = reinterpret_cast<T *>(temp_buffer);
  int *counter = reinterpret_cast<int *>(output);
#ifdef DEVICE_MALLOC_INITILIZE
  // HIP_CHECK(hipMemset(counter, 0, sizeof(int)))
  set_val<int, 0>(counter);
#endif
  (block_reduce_max_kernel<T, THREADS, ITEMS_PER_THREAD>)<<<BLOCKS, THREADS>>>(input_size, input, block_buffer, output);
}

template <typename T> void block_reduce_max_wrapper(const size_t input_size, const T *input, T *output) {
  const size_t buffer_size = block_reduce_max_buffer_size<T>(input_size);
  void *temp_buffer = nullptr;
  HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&temp_buffer), buffer_size))

  block_reduce_max_impl<T>(input_size, input, output, temp_buffer);

  HIP_CHECK(hipFree(temp_buffer))
}
} // namespace common
#endif // BLOCK_REDUCE_MAX_KERNEL_HPP