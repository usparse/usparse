//
// Created by genshen on 2021/07/06.
//

#ifndef USPARSE_HIP_USPARSE_FLAT_KERNELS_IMP_MULTI_PASS_HPP
#define USPARSE_HIP_USPARSE_FLAT_KERNELS_IMP_MULTI_PASS_HPP

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "building_config.h"

#include "flat_config.h"
#include "flat_kernels_imp_reduce.hpp"
#include "common/utils.h"

/**
 * We solve SpMV with flat method.
 *
 * @tparam WF_SIZE threads in one wavefront
 * @tparam R rounds, process R*THREADS elements in each loop.
 * @tparam REDUCE_OPTION the sum reduction method: direct or vector or vector+mem_coalescing.
 * @tparam REDUCE_VEC_SIZE threads number used for reducing sum of a row.
 * @tparam BLOCKS total blocks on one GPU (blocks in one grid).
 * @tparam THREADS threads in each block.
 * @tparam I type of data in matrix index
 * @tparam T type of data in matrix A, vector x, vector y and alpha, beta.
 * @param m rows in matrix A
 * @param alpha alpha value
 * @param beta beta value
 * @param row_offset row offset array of csr matrix A
 * @param csr_col_ind col index of csr matrix A
 * @param csr_val matrix A in csr format
 * @param x vector x
 * @param y vector y
 * @return
 */
template <int WF_SIZE, int R, int REDUCE_OPTION, int REDUCE_VEC_SIZE, int BLOCKS, int THREADS, typename I, typename T>
__global__ void spmv_flat_kernel(int m, const T alpha, const T beta, const I *__restrict__ row_offset,
                                 const I *__restrict__ break_points, const I *__restrict__ csr_col_ind,
                                 const T *__restrict__ csr_val, const T *__restrict__ x, T *__restrict__ y) {
  const int global_thread_id = threadIdx.x + THREADS * blockIdx.x;
  constexpr int global_threads_num = BLOCKS * THREADS;

  const int wf_id_in_block = blockIdx.x / WF_SIZE; // wavefront id in block
  const int block_id = blockIdx.x;                 // block id
  const int tid_in_block = threadIdx.x % THREADS;  // thread id in one block

  constexpr unsigned int shared_len = THREADS * R; // 64 * 1024 / (BLOCKS / 64) / sizeof(T); // nnz per block
  __shared__ T shared_val[shared_len];
  constexpr int nnz_per_block = THREADS * R;
  const I last_element_index = row_offset[m];

  I bp_index = block_id;
  // each block process THREADS * R elements.
  for (int k = nnz_per_block * block_id; k < last_element_index; k += BLOCKS * nnz_per_block) {
    __syncthreads();
#pragma unroll
    for (int i = 0; i < R; i++) {
      const I shared_inx = tid_in_block + i * THREADS;
      const I index = min(k + shared_inx, last_element_index - 1);
      shared_val[shared_inx] = csr_val[index] * x[csr_col_ind[index]];
    }
    __syncthreads();

    // reduce via LDS.
    const I reduce_start_row_id = min(break_points[bp_index], m);
    I reduce_end_row_id = min(break_points[bp_index + 1], m);
    // if it is the last block
    if (reduce_end_row_id == 0) {
      reduce_end_row_id = m;
    }
    // if start of the next block cuts some row.
    // or some row cross multiple blocks.
    if (row_offset[reduce_end_row_id] % nnz_per_block != 0 || reduce_end_row_id == reduce_start_row_id) {
      reduce_end_row_id = min(reduce_end_row_id + 1, m); // make sure `reduce_end_row_id` is not large than m
    }

    // reduce and store result to memory
    if (REDUCE_OPTION == FLAT_REDUCE_OPTION_VEC) {
      const I n_reduce_rows_num = reduce_end_row_id - reduce_start_row_id;
      flat_reduce_in_vector<I, T, nnz_per_block, THREADS, REDUCE_VEC_SIZE>(n_reduce_rows_num, tid_in_block, bp_index,
                                                                           reduce_start_row_id, reduce_end_row_id,
                                                                           alpha, row_offset, shared_val, y);
    } else if (REDUCE_OPTION == FLAT_REDUCE_OPTION_VEC_MEM_COALESCING) {
      const I n_reduce_rows_num = reduce_end_row_id - reduce_start_row_id;
      flat_reduce_in_vector_with_mem_coalescing<I, T, nnz_per_block, THREADS, REDUCE_VEC_SIZE>(
          n_reduce_rows_num, tid_in_block, bp_index, reduce_start_row_id, reduce_end_row_id, alpha, row_offset,
          shared_val, y);
    } else if (REDUCE_OPTION == FLAT_REDUCE_OPTION_SEGMENT_SUM) {
      __shared__ I shared_idx[THREADS];
      flat_reduce_segment_sum<I, T, nnz_per_block, R, THREADS>(tid_in_block, bp_index, reduce_start_row_id,
                                                               reduce_end_row_id, alpha, row_offset, last_element_index,
                                                               shared_idx, shared_val, y);
    } else {
      // direct reduction
      flat_reduce_direct<I, T, nnz_per_block, THREADS>(tid_in_block, bp_index, reduce_start_row_id, reduce_end_row_id,
                                                       alpha, row_offset, shared_val, y);
    }

    bp_index += BLOCKS;
  }
}

#define FLAT_KERNEL_WRAPPER(INDEX_TYPE, FP, R, REDUCE_OPTION, REDUCE_VEC_SIZE, BLOCKS, THREADS)                                    \
  (spmv_flat_kernel<__WF_SIZE__, R, REDUCE_OPTION, REDUCE_VEC_SIZE, BLOCKS, THREADS, INDEX_TYPE,                              \
                    FP>)<<<(BLOCKS), (THREADS)>>>(m, alpha, beta, rowptr, break_points, colindex, value, x, y)

#endif // USPARSE_HIP_USPARSE_FLAT_KERNELS_IMP_MULTI_PASS_HPP
