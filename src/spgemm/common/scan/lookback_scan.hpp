#ifndef LOOKBACK_SCAN_HPP
#define LOOKBACK_SCAN_HPP

#include "../check/hip_check.hpp"
#include <hip/hip_runtime.h>

namespace lookback_scan {

constexpr int X = 0;
constexpr int A = 1;
constexpr int P = 2;

constexpr size_t THREADS = 1024;
constexpr size_t ITEMS_PER_THREAD = 4;
constexpr size_t ITEMS_PER_BLOCK = ITEMS_PER_THREAD * THREADS;

using PartStateType = unsigned long long;

template <typename T> struct alignas(8) LookbackPartState {
  int flag;
  T aggregate;
};

template <typename T> __device__ void set_part_state(PartStateType *part_state, const int flag, const T aggregate) {
  struct LookbackPartState<T> tmp_state = {
    flag, aggregate
  };
  PartStateType p;
  __builtin_memcpy(&p, &tmp_state, sizeof(PartStateType));
  *part_state = p;
}

template <typename T> __device__ int get_flag(PartStateType *part_state) {
  LookbackPartState<T> tmp_state;
  __builtin_memcpy(&tmp_state, part_state, sizeof(PartStateType));
  return tmp_state.flag;
}

template <typename T> __device__ T get_aggregate(PartStateType *part_state) {
  LookbackPartState<T> tmp_state;
  __builtin_memcpy(&tmp_state, part_state, sizeof(PartStateType));
  return tmp_state.aggregate;
}

template <typename T, size_t THREADS, size_t ITEMS_PER_THREAD>
__global__ void __launch_bounds__(THREADS)
    decoupled_look_back_scan_kernel(const size_t input_size, const T *input, T *output,
                                    PartStateType *part_state_array_ptr, T *prefix_sum_array_ptr, int *id_count_ptr) {

  const int block_thread_id = threadIdx.x;
  const int block_wf_id = block_thread_id / __WF_SIZE__;
  const int wf_thread_id = block_thread_id % __WF_SIZE__;

  const int last_block_id = gridDim.x - 1;

  __shared__ int flat_block_id;
  if (block_thread_id == 0) {
    flat_block_id = atomicAdd(id_count_ptr, 1);
  }
  __syncthreads();

  const int block_item_begin = flat_block_id * THREADS * ITEMS_PER_THREAD;
  const int block_item_end = min(block_item_begin + THREADS * ITEMS_PER_THREAD, input_size);

  PartStateType *block_state_ptr = part_state_array_ptr + flat_block_id;
  T *prefix_sum_ptr = prefix_sum_array_ptr + flat_block_id;

  __shared__ T shared_reduce_sum;
  __shared__ T shared_addend;
  __shared__ T shared_scan_offset[THREADS / __WF_SIZE__];

  T local_input[ITEMS_PER_THREAD];

  if (block_thread_id == 0) {
    shared_reduce_sum = static_cast<T>(0);
    shared_addend = static_cast<T>(0);
  }
  T thread_local_sum = static_cast<T>(0);

  if (flat_block_id != last_block_id) {
#pragma unroll ITEMS_PER_THREAD
    for (int r = 0; r < ITEMS_PER_THREAD; r++) {
      const int idx = block_item_begin + THREADS * r + block_thread_id;
      const T num = input[idx];
      local_input[r] = num;
      thread_local_sum += num;
    }
  } else {
    for (int r = 0; r < ITEMS_PER_THREAD; r++) {
      const int idx = block_item_begin + THREADS * r + block_thread_id;
      if (idx < block_item_end) {
        const T num = input[idx];
        local_input[r] = num;
        thread_local_sum += num;
      }
    }
  }

  for (int i = __WF_SIZE__ >> 1; i > 0; i >>= 1) {
    thread_local_sum += __shfl_down(thread_local_sum, i, __WF_SIZE__);
  }
  if (wf_thread_id == 0) {
    atomicAdd(&shared_reduce_sum, thread_local_sum);
  }
  __syncthreads();
  if (flat_block_id == 0) {
    if (block_thread_id == 0) {
      (*prefix_sum_ptr) = shared_reduce_sum;
      __threadfence();
      set_part_state(block_state_ptr, P, shared_reduce_sum);
    }
  } else {
    // update state
    if (block_thread_id == 0) {
      set_part_state(block_state_ptr, A, shared_reduce_sum);
    }
    if (block_wf_id == 0) {
      const int track_flat_id = (flat_block_id - wf_thread_id - 1) > 0 ? (flat_block_id - wf_thread_id - 1) : 0;
      int f = get_flag<T>(part_state_array_ptr + track_flat_id);
      while (!(__any(f == P) && __all(f == P || f == A))) {
        // wait
        __threadfence();
        // update flag
        f = get_flag<T>(part_state_array_ptr + track_flat_id);
      }
      T addend = get_aggregate<T>(part_state_array_ptr + track_flat_id);
      const int last_wf_thread_id = __ffsll(__ballot(f == P)) - 1;
      if (wf_thread_id == last_wf_thread_id) {
        addend = *(prefix_sum_array_ptr + track_flat_id);
      }
      if (wf_thread_id > last_wf_thread_id) {
        addend = static_cast<T>(0);
      }
      for (int i = __WF_SIZE__ >> 1; i > 0; i >>= 1) {
        addend += __shfl_down(addend, i, __WF_SIZE__);
      }
      if (wf_thread_id == 0) {
        shared_addend = addend;
        (*prefix_sum_ptr) = shared_addend + shared_reduce_sum;
        __threadfence();
        set_part_state(block_state_ptr, P, shared_reduce_sum);
      }
    }
    __syncthreads();
  }
  __syncthreads();
  // scan
  if (flat_block_id != last_block_id) {
    for (int r = 0; r < ITEMS_PER_THREAD; r++) {
      const int idx = block_item_begin + THREADS * r + block_thread_id;
      T num = local_input[r];
      for (int i = 1; i < __WF_SIZE__; i <<= 1) {
        T tmp = __shfl_up(num, i, __WF_SIZE__);
        if (wf_thread_id >= i) {
          num += tmp;
        }
      }
      if (wf_thread_id == __WF_SIZE__ - 1) {
        shared_scan_offset[block_wf_id] = num;
      }
      __syncthreads();
      for (int i = 1; i < THREADS / __WF_SIZE__; i++) {
        if (block_wf_id >= i) {
          num += shared_scan_offset[i - 1];
        }
      }
      output[idx] = shared_addend + num;
      __syncthreads();
      if (block_thread_id == THREADS - 1) {
        shared_addend += num;
      }
    }
  } else {
    for (int r = 0; r < ITEMS_PER_THREAD; r++) {
      const int idx = block_item_begin + THREADS * r + block_thread_id;
      if (idx < block_item_end) {
        T num = local_input[r];
        for (int i = 1; i < __WF_SIZE__; i <<= 1) {
          T tmp = __shfl_up(num, i, __WF_SIZE__);
          if (wf_thread_id >= i) {
            num += tmp;
          }
        }
        if (wf_thread_id == __WF_SIZE__ - 1) {
          shared_scan_offset[block_wf_id] = num;
        }
        __syncthreads();
        for (int i = 1; i < THREADS / __WF_SIZE__; i++) {
          if (block_wf_id >= i) {
            num += shared_scan_offset[i - 1];
          }
        }
        output[idx] = shared_addend + num;
        __syncthreads();
        if (block_thread_id == THREADS - 1) {
          shared_addend += num;
        }
      }
    }
  }
}

template <typename T> size_t decoupled_look_back_scan_buffer_size(const size_t input_size) {
  static_assert(sizeof(PartStateType) == sizeof(LookbackPartState<T>), "item must less then 32bits");

  const size_t BLOCKS = input_size / ITEMS_PER_BLOCK + (input_size % ITEMS_PER_BLOCK == 0 ? 0 : 1);
  size_t buffer_size = 0;
  buffer_size += ((BLOCKS * sizeof(PartStateType)) / 256 + 1) * 256;
  buffer_size += ((BLOCKS * sizeof(T)) / 256 + 1) * 256;
  buffer_size += ((sizeof(int)) / 256 + 1) * 256;
  return buffer_size;
}

template <typename T>
void decoupled_look_back_scan_impl(const size_t input_size, const T *input, T *output, void *temp_buffer) {
  static_assert(sizeof(PartStateType) == sizeof(LookbackPartState<T>), "item must less then 32bits");

  const size_t BLOCKS = input_size / ITEMS_PER_BLOCK + (input_size % ITEMS_PER_BLOCK == 0 ? 0 : 1);

  char *ptr = reinterpret_cast<char *>(temp_buffer);
  PartStateType *part_state_array_ptr = reinterpret_cast<PartStateType *>(ptr);
  ptr += ((BLOCKS * sizeof(PartStateType)) / 256 + 1) * 256;

  T *prefix_sum_array_ptr = reinterpret_cast<T *>(ptr);
  ptr += ((BLOCKS * sizeof(T)) / 256 + 1) * 256;

  int *id_count_ptr = reinterpret_cast<int *>(ptr);
  HIP_CHECK(hipMemset(id_count_ptr, 0, sizeof(int)))

  (decoupled_look_back_scan_kernel<T, THREADS, ITEMS_PER_THREAD>)<<<BLOCKS, THREADS>>>(
      input_size, input, output, part_state_array_ptr, prefix_sum_array_ptr, id_count_ptr);
}

template <typename T> void decoupled_look_back_scan_wrapper(const size_t input_size, const T *input, T *output) {
  const size_t buffer_size = decoupled_look_back_scan_buffer_size<T>(input_size);

  void *temp_buffer = nullptr;
  HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&temp_buffer), buffer_size))

  decoupled_look_back_scan_impl<T>(input_size, input, output, temp_buffer);

  HIP_CHECK(hipFree(temp_buffer))
}
} // namespace lookback_scan

#endif // LOOKBACK_SCAN_HPP