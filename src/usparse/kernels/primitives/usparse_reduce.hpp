//
// Created by genshen on 2025/7/17.
//

#ifndef USPARSE_KERNELS__USPARSE_REDUCE_H
#define USPARSE_KERNELS__USPARSE_REDUCE_H

// source from: https://git.hpcer.dev/PRA/spmv/spmv-acc/-/blob/main/src/acc/common/utils.h
// and https://git.hpcer.dev/PRA/spmv/spmv-acc/-/blob/main/src/acc/hip-csr-adaptive-plus/csr_adaptive_plus_spmv_imp.inl
// and https://git.hpcer.dev/PRA/spmv/spmv-acc/-/blob/main/src/acc/hip-line-enhance/line_enhance_reduce.hpp

namespace usparse {
  // local shift can reduce data inside a vector to the first thread in the vector.
  template <typename I, typename T, int VECTOR_SIZE>
  __device__ __forceinline__ void line_enhance_vec_local_shift(T &data) {
    // reduce from lanes in a vector
    if (VECTOR_SIZE > 1) { // in fact, this branch is unnecessary.
#pragma unroll
      for (int i = VECTOR_SIZE >> 1; i > 0; i >>= 1) {
        data += __shfl_down(data, i, VECTOR_SIZE);
        // or use: __shfl_down_sync(0xffffffff, i, VECTOR_SIZE);
      }
    }
  }

  /**
   * performance reduce inside a block.
   * @tparam I index type
   * @tparam T data type
   * @tparam THREADS_IN_BLOCK number of threads inside the block.
   * @tparam WF_SIZE number of threads inside the wavefront/warp.
   * @param sum the input data. Each thread contributs one value.
   */
  template <typename I, typename T, int THREADS_IN_BLOCK, int WF_SIZE>
  __device__ __forceinline__ T line_enhance_block_reduce(T &sum, const I tid_in_block) {
    // 1: recuce inside wavefront first. only the first thread in each wave has value.
    line_enhance_vec_local_shift<I, T, WF_SIZE>(sum);

    // 2: reduce among wavefronts inside a block.
    constexpr int wavefronts_num = THREADS_IN_BLOCK / WF_SIZE;
    __shared__ T shared_val[wavefronts_num];
    const int wf_id_in_block = tid_in_block / WF_SIZE;
    const int tid_in_wf = tid_in_block % WF_SIZE;
    if (tid_in_wf == 0) {
      shared_val[wf_id_in_block] = sum;
    }
    __syncthreads();
    // use a wavefront to reduce from LDS is encough
    T reduced_sum = 0;
    if (wf_id_in_block == 0 && tid_in_wf < wavefronts_num) {
      reduced_sum = shared_val[tid_in_wf];
    }
    // number of `wavefronts_num` threads as a vector for reduction.
    line_enhance_vec_local_shift<I, T, wavefronts_num>(reduced_sum);
    return reduced_sum;
  }
} // namespace usparse

#endif // USPARSE_KERNELS__USPARSE_REDUCE_H
