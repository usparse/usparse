//
// Created by chu genshen on 2021/9/13.
//

#ifndef USPARSE_HIP_USPARSE_FLAT_KERNELS_IMP_ONE_PASS_HPP
#define USPARSE_HIP_USPARSE_FLAT_KERNELS_IMP_ONE_PASS_HPP

#include "flat_config.h"
#include "flat_kernels_imp_reduce.hpp"
#include "building_config.h"

#include "common/global_mem_ops.h"
#include "common/utils.h"

template <int WF_SIZE, int R, int REDUCE_OPTION, int REDUCE_VEC_SIZE, int THREADS, typename SHARE_MEM_INDEX_TYPE,
          typename BP_INDEX_TYPE, typename BP_DATA_TYPE, typename SCALE_DATA_TYPE, typename A_TYPE,
          typename COMPUTE_MULTI_TYPE, typename COMPUTE_SUM_TYPE, typename X_TYPE, typename Y_TYPE, typename NNZ_TYPE,
          typename ROW_TYPE, typename COLUMN_TYPE>
__global__ __launch_bounds__(THREADS) void csrmv_flat_one_pass_kernel(
    ROW_TYPE m, const SCALE_DATA_TYPE alpha, const SCALE_DATA_TYPE beta, const NNZ_TYPE *__restrict__ row_offset,
    const BP_DATA_TYPE *__restrict__ break_points, const COLUMN_TYPE *__restrict__ csr_col_ind,
    const A_TYPE *__restrict__ csr_val, const X_TYPE *__restrict__ x, Y_TYPE *__restrict__ y) {
  const int block_id = blockIdx.x;                // block id
  const int tid_in_block = threadIdx.x % THREADS; // thread id in one block

  constexpr int nnz_per_block = THREADS * R;
  __shared__ COMPUTE_MULTI_TYPE shared_val[nnz_per_block];
  const NNZ_TYPE last_element_index = row_offset[m];

  BP_INDEX_TYPE bp_index = static_cast<BP_INDEX_TYPE>(block_id);
  // each block process THREADS * R elements.
  NNZ_TYPE k = static_cast<NNZ_TYPE>(nnz_per_block) * static_cast<NNZ_TYPE>(block_id);

#pragma unroll
  for (int i = 0; i < R; i++) {
    const SHARE_MEM_INDEX_TYPE shared_inx = tid_in_block + i * THREADS;
    const NNZ_TYPE index = min(k + static_cast<NNZ_TYPE>(shared_inx), last_element_index - 1);
    shared_val[shared_inx] =
        static_cast<COMPUTE_MULTI_TYPE>(csr_val[index]) * static_cast<COMPUTE_MULTI_TYPE>(x[csr_col_ind[index]]);
  }
  __syncthreads();

  // reduce via LDS.
  const ROW_TYPE reduce_start_row_id = min(break_points[bp_index], m);
  // note: if it is the last block, reduce_end_row_id is m (this is determined in the preprocessing stage)
  ROW_TYPE reduce_end_row_id = min(break_points[bp_index + 1], m);

  // if start of the next block cuts some row.
  // or some row cross multiple blocks.
  // what if it has a very long row? which means `reduce_start_row_id == reduce_end_row_id`.
  if (row_offset[reduce_end_row_id] % nnz_per_block != 0 || reduce_end_row_id == reduce_start_row_id) {
    reduce_end_row_id = min(reduce_end_row_id + 1, m); // make sure `reduce_end_row_id` is not larger than m
  }

  // reduce and store result to memory
  if (REDUCE_OPTION == FLAT_REDUCE_OPTION_VEC) {
    const ROW_TYPE n_reduce_rows_num = reduce_end_row_id - reduce_start_row_id;
    flat_reduce_in_vector<BP_INDEX_TYPE, COMPUTE_MULTI_TYPE, SHARE_MEM_INDEX_TYPE, NNZ_TYPE, ROW_TYPE, SCALE_DATA_TYPE,
                          COMPUTE_SUM_TYPE, Y_TYPE, nnz_per_block, THREADS, REDUCE_VEC_SIZE>(
        n_reduce_rows_num, tid_in_block, bp_index, reduce_start_row_id, reduce_end_row_id, alpha, row_offset,
        shared_val, y);
  } else if (REDUCE_OPTION == FLAT_REDUCE_OPTION_VEC_MEM_COALESCING) {
    const ROW_TYPE n_reduce_rows_num = reduce_end_row_id - reduce_start_row_id;
    flat_reduce_in_vector_with_mem_coalescing<BP_INDEX_TYPE, COMPUTE_MULTI_TYPE, SHARE_MEM_INDEX_TYPE, NNZ_TYPE,
                                              ROW_TYPE, SCALE_DATA_TYPE, COMPUTE_SUM_TYPE, Y_TYPE, nnz_per_block,
                                              THREADS, REDUCE_VEC_SIZE>(n_reduce_rows_num, tid_in_block, bp_index,
                                                                        reduce_start_row_id, reduce_end_row_id, alpha,
                                                                        row_offset, shared_val, y);
  } else if (REDUCE_OPTION == FLAT_REDUCE_OPTION_SEGMENT_SUM) {
    __shared__ ROW_TYPE shared_idx[THREADS];
    flat_reduce_segment_sum<BP_INDEX_TYPE, COMPUTE_MULTI_TYPE, SHARE_MEM_INDEX_TYPE, NNZ_TYPE, ROW_TYPE,
                            SCALE_DATA_TYPE, COMPUTE_SUM_TYPE, Y_TYPE, nnz_per_block, R, THREADS>(
        tid_in_block, bp_index, reduce_start_row_id, reduce_end_row_id, alpha, row_offset, last_element_index,
        shared_idx, shared_val, y);
  } else {
    // direct reduction
    flat_reduce_direct<BP_INDEX_TYPE, COMPUTE_MULTI_TYPE, SHARE_MEM_INDEX_TYPE, NNZ_TYPE, ROW_TYPE, SCALE_DATA_TYPE,
                       COMPUTE_SUM_TYPE, Y_TYPE, nnz_per_block, THREADS>(
        tid_in_block, bp_index, reduce_start_row_id, reduce_end_row_id, alpha, row_offset, shared_val, y);
  }
}

#endif // USPARSE_HIP_USPARSE_FLAT_KERNELS_IMP_ONE_PASS_HPP
