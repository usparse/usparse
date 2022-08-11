#pragma once

#include "common/scan_kernel.hpp"
#include "common/reduction_kernel.hpp"
#include "building_config.h"

template <typename T, typename I,int DIM_X, int DIM_Y>
__global__ void nnz_kernel(const size_t m, const size_t n, const T *A, const size_t lda, I *ptr_offset_A)
{
    static constexpr T s_zero = {};

    int thread_id = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x;
    int tx        = thread_id % DIM_X;
    int ty        = thread_id / DIM_X;
    int ind       = hipBlockIdx_x * DIM_X * 4 + tx;
    int n_tail    = n % (4 * DIM_Y);
    int col       = ty * 4;
    int res_A[4];

    __shared__ I sdata[DIM_X * 4 * DIM_Y];

    for(int k = 0; k < 4; ++k)
    {
        res_A[k] = 0;
    }

    for(col = ty * 4; col < (n - n_tail); col += 4 * DIM_Y)
    {
        for(int k = 0; k < 4; ++k)
        {
            if(ind + k * DIM_X < m)
            {
                    for(int j = 0; j < 4; ++j)
                    {
                        if(A[(ind + k * DIM_X) * lda + col + j] != s_zero)
                            res_A[k] += 1;
                    }
            }
        }
    }

    if(n_tail > 0)
    {
        for(int k = 0; k < 4; ++k)
        {
            if(ind + k * DIM_X < m)
            {
                for(int j = 0; j < 4; ++j)
                {
                    if(col + j < n)
                    {
                        res_A[k] += (A[(ind + k * DIM_X) * lda + col + j] != s_zero) ? 1 : 0;
                    }
                }
            }
        }
    }

    for(int k = 0; k < 4; ++k)
    {
        sdata[tx + k * DIM_X + ty * DIM_X * 4] = res_A[k];
    }

    __syncthreads();

    ind = hipBlockIdx_x * DIM_X * 4 + thread_id;
    if(thread_id < DIM_X * 4)
    {
        for(int j = 1; j < DIM_Y; j++)
        {
            sdata[thread_id] += sdata[thread_id + DIM_X * 4 * j];
        }

        if(ind < m)
        {
            ptr_offset_A[ind] = sdata[thread_id];
        }
    }
}

template <typename T, typename I,int THREADS, int LOOP_UNROLL>
__global__ void nnz_kernel_long_row(const size_t m, const size_t n, const T *A, const size_t lda,
                                    I *ptr_offset_A) {
  constexpr T s_zero = {};
  constexpr int WAVES_IN_BLOCK = THREADS / __WF_SIZE__;

  const int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  const int bid = hipBlockIdx_x;
  const int tid_in_wf = tid % __WF_SIZE__;
  const int tid_in_block = tid % THREADS;
  const int wave_id_in_block = tid_in_block / __WF_SIZE__;

  const int &row_id = bid;
  const I start_index = n * bid;
  const I end_index = n * (bid + 1);

  I X[LOOP_UNROLL];
  #pragma unroll
  for(int i=0; i < LOOP_UNROLL; i++) {
    X[i] = static_cast<I>(0);
  }

  // const int loop_count = n / (LOOP_UNROLL * THREADS);
  for(int k = start_index + LOOP_UNROLL * tid_in_block; k < end_index; k += LOOP_UNROLL*THREADS) {
    #pragma unroll
    for(int i=0; i < LOOP_UNROLL; i++) {
        const T val = A[k + i];
        if (val != s_zero) {
            X[i]++;
        }
    }
  }

  #pragma unroll
  for(int i = 1; i < LOOP_UNROLL; i++) {
    X[0] += X[i];
  }

  _type_vec1<I> nnz;
  nnz.set_v(X[0]);

  // perform reduction
  __shared__ I temp_mem[WAVES_IN_BLOCK];
  (nnz.template block_wf_reduce<__WF_SIZE__, THREADS, WAVES_IN_BLOCK>)(tid_in_wf, tid_in_block, wave_id_in_block, (_type_vec1<I> *) temp_mem);
  if(tid_in_block == 0) {
    ptr_offset_A[bid] = nnz.data;
  }
}
