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

template <int BREAK_STRIDE, int BLOCKS, typename I>
__global__ void pre_calc_break_point(const I *__restrict__ row_ptr, const I m, I *__restrict__ break_points,
                                     const I bp_len) {
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  const int global_threads_num = blockDim.x * gridDim.x;

  constexpr I break_stride = BREAK_STRIDE;
  if (global_thread_id == 0) {
    break_points[0] = 0; // start row of the block 0 and the first round.
  }

  for (int i = global_thread_id; i < m; i += global_threads_num) {
    // for first element of row i and row i+1, they belong to different blocks.
    if (row_ptr[i] / break_stride != row_ptr[i + 1] / break_stride) { // fixme: step may be not 1
      // record the row id of the first element in the block.
      // note: a row can cross multiple blocks.
      for (int j = row_ptr[i] / break_stride + 1; j <= row_ptr[i + 1] / break_stride; j++) {
        break_points[j] = i;
      }
      if (row_ptr[i + 1] % break_stride == 0) {
        break_points[row_ptr[i + 1] / break_stride] += 1;
      }
    }
  }
}

// a more simple kernel for calculation break-point.
template <int BREAK_STRIDE, int BLOCKS, typename I>
__global__ void pre_calc_break_point_v2(const I *__restrict__ row_ptr, const I m, I *__restrict__ break_points,
                                        const I bp_len) {
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  const int global_threads_num = blockDim.x * gridDim.x;

  constexpr I break_stride = BREAK_STRIDE;

  for (int i = global_thread_id; i < m; i += global_threads_num) {
    int p1 = row_ptr[i] / break_stride;
    if (row_ptr[i] % break_stride != 0) {
      p1++;
    }
    const int p2 = (row_ptr[i + 1] - 1) / break_stride;
    for (int j = p1; j <= p2; j++) {
      break_points[j] = i;
    }
  }
}

#endif // USPARSE_HIP_USPARSE_FLAT_KERNELS_IMP_PRE_CALC_HPP
