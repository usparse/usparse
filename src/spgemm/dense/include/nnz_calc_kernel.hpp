#pragma once

#include "types.hpp"
#include "utils_kernel.hpp"
#include "common/reduction.hpp"

template <int DIM_X, int DIM_Y>
__global__ void nnz_kernel(const int m, const int n, const csrValueType *A, const int lda, csrIndexType *ptr_offset_A)
{
    static constexpr csrValueType s_zero = {};

    int thread_id = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x;
    int tx        = thread_id % DIM_X;
    int ty        = thread_id / DIM_X;
    int ind       = hipBlockIdx_x * DIM_X * 4 + tx;
    int n_tail    = n % (4 * DIM_Y);
    int col       = ty * 4;
    int res_A[4];

    __shared__ int sdata[DIM_X * 4 * DIM_Y];

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

template <int THREADS, int WAVE_SIZE, int LOOP_UNROLL>
__global__ void nnz_kernel_long_row(const int m, const int n, const csrValueType *A, const int lda,
                                    csrIndexType *ptr_offset_A) {
  constexpr csrValueType s_zero = {};
  constexpr int WAVES_IN_BLOCK = THREADS / WAVE_SIZE;

  const int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  const int bid = hipBlockIdx_x;
  const int tid_in_wf = tid % WAVE_SIZE;
  const int tid_in_block = tid % THREADS;
  const int wave_id_in_block = tid_in_block / WAVE_SIZE;

  const int &row_id = bid;
  const csrIndexType start_index = n * bid;
  const csrIndexType end_index = n * (bid + 1);

  csrIndexType X[LOOP_UNROLL];
  #pragma unroll
  for(int i=0; i < LOOP_UNROLL; i++) {
    X[i] = static_cast<csrIndexType>(0);
  }

  // const int loop_count = n / (LOOP_UNROLL * THREADS);
  for(int k = start_index + LOOP_UNROLL * tid_in_block; k < end_index; k += LOOP_UNROLL*THREADS) {
    #pragma unroll
    for(int i=0; i < LOOP_UNROLL; i++) {
        const csrValueType val = A[k + i];
        if (val != s_zero) {
            X[i]++;
        }
    }
  }

  #pragma unroll
  for(int i = 1; i < LOOP_UNROLL; i++) {
    X[0] += X[i];
  }

  _type_vec1<csrIndexType> nnz;
  nnz.set_v(X[0]);

  // perform reduction
  __shared__ csrIndexType temp_mem[WAVES_IN_BLOCK];
  (nnz.block_wf_reduce<WAVE_SIZE, THREADS, WAVES_IN_BLOCK>)(tid_in_wf, tid_in_block, wave_id_in_block,
                                                            (_type_vec1<csrIndexType> *)temp_mem);
  if(tid_in_block == 0) {
    ptr_offset_A[bid] = nnz.data;
  }
}

void nnz_kernel_wrapper(const int m, const int n, const csrValueType* A, const int ld, csrIndexType* ptr_offset_A) {
    // for short row
    if(n <= 1024) {
        static constexpr int NNZ_DIM_X = 64;
        static constexpr int NNZ_DIM_Y = 16;
        int blocks= (m - 1) / (NNZ_DIM_X * 4) + 1;
        // blocks = (m - 1) / (NNZ_DIM_X) + 1;
        dim3 k_grid(blocks);
        dim3 k_threads(NNZ_DIM_X, NNZ_DIM_Y);
        nnz_kernel<NNZ_DIM_X, NNZ_DIM_Y><<<k_grid, k_threads>>>(m, n, A,ld, ptr_offset_A+1);
    } else { // long row: one block for one row
        constexpr int THREADS = 64;
        const int BLOCKS = m;
        constexpr int __WF_SIZE__ = 64;
        if(n % 4 == 0) {
            (nnz_kernel_long_row<THREADS, __WF_SIZE__, 4>)<<<BLOCKS, THREADS>>>(m, n, A, ld, ptr_offset_A+1);
        } else {
            (nnz_kernel_long_row<THREADS, __WF_SIZE__, 1>)<<<BLOCKS, THREADS>>>(m, n, A, ld, ptr_offset_A+1);
        }
    }
}

template<int WF_SIZE, int THREADS, int R, int VEC_SIZE>
void reduction_kernel_warpper(const int m, const int n, const csrValueType* A, const int ld, csrIndexType* ptr_offset_A,
    int *temp_buffer, const int BLOCKS, const int LEN) {
   // reduce than fan
    prefix_sum_kernel<csrIndexType, VEC_SIZE, R, THREADS, WF_SIZE><<<BLOCKS, THREADS>>>(LEN, ptr_offset_A+1, temp_buffer);
    buffer_reduce_kernel<csrIndexType, VEC_SIZE, THREADS, WF_SIZE><<<1, THREADS>>>(BLOCKS, temp_buffer);
    reduction_addend_kernel<csrIndexType, R, THREADS, WF_SIZE><<<BLOCKS, THREADS>>>(m + 1, ptr_offset_A+1, temp_buffer);
}

template<int WF_SIZE, int THREADS, int R>
size_t nnz_calc(const int m, const int n, const csrValueType* A, const int ld, csrIndexType* ptr_offset_A, int *temp_buffer)
{
    const size_t LEN = m + 1;
    const size_t BLOCKS = LEN / (THREADS * R) + (LEN % (THREADS * R) == 0 ? 0 : 1);
    if(temp_buffer == nullptr) {
        const size_t BUFFER_SIZE = (BLOCKS) * sizeof(int);
        return BUFFER_SIZE;
    }
    (reduction_kernel_warpper<WF_SIZE, THREADS, R, 32>)(m, n, A, ld, ptr_offset_A, temp_buffer, BLOCKS, LEN);
    return 0;
}
