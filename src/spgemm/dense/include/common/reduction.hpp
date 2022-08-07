//
// Created by genshen on 2022/6/5.
//

#ifndef MISA_MD_HIP_REDUCE_HPP
#define MISA_MD_HIP_REDUCE_HPP

#include <hip/hip_runtime.h>

// register and __shfl_down based wavefront reduction
template <typename T, int WF_SIZE>
__device__ __forceinline__ void SHFL_DOWN_WF_REDUCE(T &total_sum, const T local_sum) {
  total_sum += local_sum;
  if (WF_SIZE > 32) {
    total_sum += __shfl_down(total_sum, 32, WF_SIZE);
  }
  if (WF_SIZE > 16) {
    total_sum += __shfl_down(total_sum, 16, WF_SIZE);
  }
  if (WF_SIZE > 8) {
    total_sum += __shfl_down(total_sum, 8, WF_SIZE);
  }
  if (WF_SIZE > 4) {
    total_sum += __shfl_down(total_sum, 4, WF_SIZE);
  }
  if (WF_SIZE > 2) {
    total_sum += __shfl_down(total_sum, 2, WF_SIZE);
  }
  if (WF_SIZE > 1) {
    total_sum += __shfl_down(total_sum, 1, WF_SIZE);
  }
}

template <typename T> struct _type_vec1 {
  T data = static_cast<T>(0);

  __device__ __forceinline__ void set_v(const _type_vec1<T> vec1) { data = vec1.data; }
  __device__ __forceinline__ void set_v(const T val) { data = val; }

  __device__ __forceinline__ void inc() { data += static_cast<T>(1); }

  __device__ __forceinline__ void inc(const T detal) { data += static_cast<T>(detal); }

  template <typename I> __device__ __forceinline__ void store_to(_type_vec1<T> *ptr, I offset) {
     ptr[offset] = *this;
  }

 template <int WAVE_SIZE>
  __device__ __forceinline__ void wf_reduce() {
    T t0_sum = static_cast<T>(0);
    SHFL_DOWN_WF_REDUCE<T, WAVE_SIZE>(t0_sum, data);
    data = t0_sum;
  }

  /**
   *  reduce data in register of all threads in current block into thread 0 of current block.
   */
  template <int WAVE_SIZE, int THREADS_IN_BLOCK, int WAVES_IN_BLOCK>
  __device__ __forceinline__ void block_wf_reduce(const int tid_in_wf, const int tid_in_block,
                                                  const int wave_id_in_block, _type_vec1<T> *temp_mem) {
    wf_reduce<WAVE_SIZE>();
    // reduction in shared memory
    if (tid_in_wf == 0) {
      temp_mem[wave_id_in_block].data = this->data;
    }

    data = static_cast<T>(0);
    __syncthreads(); // wait shared memory writing finishes.
    if (wave_id_in_block == 0) {
#pragma unroll
      for (int i = tid_in_block; i < WAVES_IN_BLOCK; i += WAVE_SIZE) {
        data += temp_mem[i].data;
      }
      wf_reduce<WAVE_SIZE>(); // we assume that the threads in block is not bigger than WAVE_SIZE*WAVE_SIZE
    }
  }
};

#endif // MISA_MD_HIP_REDUCE_HPP
