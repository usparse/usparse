//
// Created by chu genshen on 2025/12/25.
//

#ifndef USPARSE_HIP_USPARSE_FLAT_KERNELS_IMP_PRE_CALC_HPP
#define USPARSE_HIP_USPARSE_FLAT_KERNELS_IMP_PRE_CALC_HPP

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "building_config.h"

#include "flat_config.h"

template <typename ROW_TYPE, typename NNZ_TYPE, typename BP_TYPE, NNZ_TYPE BREAK_STRIDE>
__global__ __launch_bounds__(512) void pre_calc_break_point(const NNZ_TYPE *__restrict__ row_ptr, const ROW_TYPE m,
                                                            BP_TYPE *__restrict__ break_points,
                                                            const NNZ_TYPE break_points_len) {
  const ROW_TYPE global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  const ROW_TYPE global_threads_num = blockDim.x * gridDim.x;

  constexpr NNZ_TYPE break_stride = BREAK_STRIDE;
  if (global_thread_id == 0) {
    break_points[0] = static_cast<BP_TYPE>(0); // start row of the block 0 and the first round.
    // the end row of the last HIP block in the flat kernel is m.
    break_points[break_points_len - 1] = static_cast<BP_TYPE>(m);
  }

  for (ROW_TYPE i = global_thread_id; i < m; i += global_threads_num) {
    // for first element of row i and row i+1, they belong to different blocks.
    const NNZ_TYPE block_id_1 = row_ptr[i] / break_stride;
    const NNZ_TYPE block_id_2 = row_ptr[i + 1] / break_stride;
    if (block_id_1 != block_id_2) { // fixme: step may be not 1
      // record the row id of the first element in the block.
      // note: a row can cross multiple blocks.
      for (NNZ_TYPE j = block_id_1 + 1; j <= block_id_2; j++) {
        break_points[j] = i;
      }
      if (row_ptr[i + 1] % break_stride == 0) {
        break_points[block_id_2] += 1;
      }
    }
  }
}

/**
 * a more simple kernel for calculation break-point.
 * @tparam I index type of matrix row or column
 * or break point value type (which has the same type of the matrix row, because it records the starting row of each
 * hip-block).
 * @tparam J index type of nnz
 * @tparam BREAK_STRIDE the nnz in wach block
 */
template <typename ROW_TYPE, typename NNZ_TYPE, typename BP_TYPE, NNZ_TYPE BREAK_STRIDE>
__global__ __launch_bounds__(512) void pre_calc_break_point_v2(const NNZ_TYPE *__restrict__ row_ptr, const ROW_TYPE m,
                                                               BP_TYPE *__restrict__ break_points,
                                                               const NNZ_TYPE break_points_len) {
  const ROW_TYPE global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  const ROW_TYPE global_threads_num = blockDim.x * gridDim.x;
  constexpr NNZ_TYPE break_stride = BREAK_STRIDE;

  if (global_thread_id == 0) {
    // the end row of the last HIP block in the flat kernel is m.
    break_points[break_points_len - 1] = static_cast<BP_TYPE>(m);
  }

  for (ROW_TYPE i = global_thread_id; i < m; i += global_threads_num) {
    NNZ_TYPE p1 = row_ptr[i] / break_stride;
    if (row_ptr[i] % break_stride != 0) {
      p1++;
    }
    const NNZ_TYPE p2 = (row_ptr[i + 1] - 1) / break_stride;
    for (NNZ_TYPE j = p1; j <= p2; j++) {
      break_points[j] = static_cast<BP_TYPE>(i);
    }
  }
}

#endif // USPARSE_HIP_USPARSE_FLAT_KERNELS_IMP_PRE_CALC_HPP
