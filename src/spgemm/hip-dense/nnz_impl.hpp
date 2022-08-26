#pragma once

#include "nnz_kernel.hpp"
#include "building_config.h"

template <typename T, typename I>
void nnz_kernel_wrapper(const size_t m, const size_t n, const T* A, const size_t ld, I* ptr_offset_A) {
    // for short row
    if(n <= 1024) {
        static constexpr int NNZ_DIM_X = 64;
        static constexpr int NNZ_DIM_Y = 16;
        int blocks= (m - 1) / (NNZ_DIM_X * 4) + 1;
        // blocks = (m - 1) / (NNZ_DIM_X) + 1;
        dim3 k_grid(blocks);
        dim3 k_threads(NNZ_DIM_X, NNZ_DIM_Y);
        nnz_kernel<T,I,NNZ_DIM_X, NNZ_DIM_Y><<<k_grid, k_threads>>>(m, n, A,ld, ptr_offset_A+1);
    } else { // long row: one block for one row
        constexpr int THREADS = 64;
        const int BLOCKS = m;
        if(n % 4 == 0) {
            (nnz_kernel_long_row<T,I,THREADS, 4>)<<<BLOCKS, THREADS>>>(m, n, A, ld, ptr_offset_A+1);
        } else {
            (nnz_kernel_long_row<T,I,THREADS, 1>)<<<BLOCKS, THREADS>>>(m, n, A, ld, ptr_offset_A+1);
        }
    }
}

template<typename T, typename I,int THREADS, int R, int VEC_SIZE>
void reduction_kernel_warpper(const size_t m, const size_t n, I* ptr_offset_A,
    I *temp_buffer, const int BLOCKS, const int LEN) {
   // reduce than fan
    prefix_sum_kernel<I, VEC_SIZE, R, THREADS, __WF_SIZE__><<<BLOCKS, THREADS>>>(LEN, ptr_offset_A+1, temp_buffer);
    buffer_reduce_kernel<I, VEC_SIZE, THREADS, __WF_SIZE__><<<1, THREADS>>>(BLOCKS, temp_buffer);
    reduction_addend_kernel<I, R, THREADS, __WF_SIZE__><<<BLOCKS, THREADS>>>(m + 1, ptr_offset_A+1, temp_buffer);
}

template<typename T, typename I, int THREADS, int R>
size_t nnz_calc(const size_t m, const size_t n, I* ptr_offset_A, I *temp_buffer)
{
    const size_t LEN = m + 1;
    const size_t BLOCKS = LEN / (THREADS * R) + (LEN % (THREADS * R) == 0 ? 0 : 1);
    if(temp_buffer == nullptr) {
        const size_t BUFFER_SIZE = (BLOCKS) * sizeof(I);
        return BUFFER_SIZE;
    }
    (reduction_kernel_warpper<T,I,THREADS, R, 32>)(m, n, ptr_offset_A, temp_buffer, BLOCKS, LEN);
    return 0;
}