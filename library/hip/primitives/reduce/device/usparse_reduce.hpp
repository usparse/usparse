//
// Created by genshen on 2025/7/17.
//

#ifndef USPARSE_KERNELS__USPARSE_REDUCE_H
#define USPARSE_KERNELS__USPARSE_REDUCE_H

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <hip/hip_cooperative_groups.h>

#include "include/usparse_config.hpp"

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

  namespace device {
    /**
     * Each thread contributes one value in register. this functionn reduce them and store reduced result to device
     * memory specified by  @param dev_result
     * @param thread_value an value in register in the thread. Note: value of @param thread_value may change during this
     * function.
     * @param d_temp_buffer device temp buffer memory used in reduction.
     * @param dev_result the final result stored in device memory.
     */
    template <typename INDEX_TYPE, typename DATA_TYPE, int THREADS>
    __device__ __forceinline__ void reduce_using_cooperative_groups(cooperative_groups::grid_group &grid,
                                                                    DATA_TYPE &thread_value, DATA_TYPE *d_temp_buffer,
                                                                    DATA_TYPE *dev_result) {
      // block level reduce:
      // reduce value in a block into the first thread of the block.
      const int tid_in_block = threadIdx.x;
      DATA_TYPE block_sum =
          usparse::line_enhance_block_reduce<int, DATA_TYPE, THREADS, usparse::USPARSE_WAVEFRONT_SIZE>(thread_value,
                                                                                                       tid_in_block);

      if (tid_in_block == 0) {
        d_temp_buffer[blockIdx.x] = block_sum;
      }

      grid.sync();
      // second level reduction
      DATA_TYPE local_sum = static_cast<DATA_TYPE>(0);
      if (blockIdx.x == 0) {
        if (tid_in_block < gridDim.x) {
          local_sum = d_temp_buffer[tid_in_block];
        }

        DATA_TYPE final_sum =
            usparse::line_enhance_block_reduce<int, DATA_TYPE, THREADS, usparse::USPARSE_WAVEFRONT_SIZE>(local_sum,
                                                                                                         tid_in_block);
        if (tid_in_block == 0) {
          dev_result[0] = final_sum;
        }
      }
    }
  } // namespace device
} // namespace usparse

#endif // USPARSE_KERNELS__USPARSE_REDUCE_H
