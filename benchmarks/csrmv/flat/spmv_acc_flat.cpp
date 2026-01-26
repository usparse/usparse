//
// Created by reget on 2021/11/21.
// this file is copied and modified from level2/csrmv_kernels/flat/csrmv_flat.cpp for adding timer code.
//

#ifndef SPMV_ACC_BENCHMARK_SPMV_ACC_FLAT_HPP
#define SPMV_ACC_BENCHMARK_SPMV_ACC_FLAT_HPP

#include <iostream>

#include "../utils/benchmark_time.h"
#include "../utils/timer_utils.h"
#include "benchmark_config.h"
#include "common/macros.h"
#include "level2/csrmv_kernels/flat/flat_config.h"
#include "level2/csrmv_kernels/flat/csrmv_flat.h"
#include "level2/csrmv_kernels/flat/flat_kernels_imp_multi_pass.hpp"
#include "level2/csrmv_kernels/flat/flat_kernels_imp_one_pass.hpp"
#include "level2/csrmv_kernels/flat/flat_kernels_imp_pre_calc.hpp"
#include "spmv_acc_flat.h"
#include "timer/timer.h"

template <typename INDEX_TYPE, typename DATA_TYPE, int R, int REDUCE_OPTION, int REDUCE_VEC_SIZE, int BLOCKS,
          int THREADS_PER_BLOCK>
inline void flat_multi_pass_sparse_spmv(int trans, const DATA_TYPE alpha, const DATA_TYPE beta, int m, int n, int nnz,
                                        const int *rowptr, const int *colindex, const DATA_TYPE *value,
                                        const DATA_TYPE *x, DATA_TYPE *y, BenchmarkTime *bmt) {
  hip::timer::event_timer pre_timer, calc_timer;
  pre_timer.start();
  int *break_points;
  // the nnz is rowptr[m], in one round, it can process about `blocks * R * threads_per_block` nnz.
  const int total_rounds =
      nnz / (R * THREADS_PER_BLOCK * BLOCKS) + (nnz % (R * THREADS_PER_BLOCK * BLOCKS) == 0 ? 0 : 1);
  // each round and each block both have a break point.
  const int break_points_len = total_rounds * BLOCKS + 1;
  hipMalloc((void **)&break_points, break_points_len * sizeof(int));
  hipMemset(break_points, 0, break_points_len * sizeof(int));
  (pre_calc_break_point<INDEX_TYPE, INDEX_TYPE, INDEX_TYPE, R * THREADS_PER_BLOCK>)<<<1024, 512>>>(rowptr, m, break_points);
  pre_timer.stop();

  calc_timer.start();
  {
    using SCALE_DATA_TYPE = DATA_TYPE;
    using A_TYPE = DATA_TYPE;
    using X_TYPE = DATA_TYPE;
    using Y_TYPE = DATA_TYPE;
    using NNZ_TYPE = INDEX_TYPE;
    using ROW_TYPE = INDEX_TYPE;
    using COLUMN_TYPE = INDEX_TYPE;

    using SHARE_MEM_INDEX_TYPE = INDEX_TYPE;
    using BP_INDEX_TYPE = INDEX_TYPE;
    using BP_DATA_TYPE = ROW_TYPE;
    using COMPUTE_MULTI_TYPE = DATA_TYPE;
    using COMPUTE_SUM_TYPE = DATA_TYPE;

    (csrmv_flat_multi_pass_kernel<__WF_SIZE__, R, REDUCE_OPTION, REDUCE_VEC_SIZE, BLOCKS, THREADS_PER_BLOCK,
                                  SHARE_MEM_INDEX_TYPE, BP_INDEX_TYPE, BP_DATA_TYPE, SCALE_DATA_TYPE, A_TYPE,
                                  COMPUTE_MULTI_TYPE, COMPUTE_SUM_TYPE, X_TYPE, Y_TYPE, NNZ_TYPE, ROW_TYPE,
                                  COLUMN_TYPE>)<<<(BLOCKS), (THREADS_PER_BLOCK)>>>(m, alpha, beta, rowptr, break_points,
                                                                                   colindex, value, x, y);
  }
  calc_timer.stop(true);
  if (bmt != nullptr) {
    bmt->set_time(pre_timer.time_use, calc_timer.time_use, 0.);
  }
}

template <typename INDEX_TYPE, typename DATA_TYPE, int FLAT_PRE_CALC_BP_KERNEL_VERSION>
void one_pass_flat_sparse_spmv(int trans, const DATA_TYPE alpha, const DATA_TYPE beta,
                               const csr_desc<INDEX_TYPE, DATA_TYPE> h_csr_desc,
                               const csr_desc<INDEX_TYPE, DATA_TYPE> d_csr_desc, const DATA_TYPE *x, DATA_TYPE *y,
                               BenchmarkTime *bmt) {
  using Y_TYPE = DATA_TYPE;
  using NNZ_TYPE = INDEX_TYPE;
  using ROW_TYPE = INDEX_TYPE;
  using COLUMN_TYPE = INDEX_TYPE;
  using BP_DATA_TYPE = ROW_TYPE;
  using BP_INDEX_TYPE = INDEX_TYPE;

  hip::timer::event_timer pre_timer, calc_timer;
  auto d_csr_mat_desc = usparse::csr_mat_descr<DATA_TYPE, NNZ_TYPE, ROW_TYPE, COLUMN_TYPE>(d_csr_desc);

  pre_timer.start();
  csrmv_flat_mtx_info info;
  flat_analyse<DATA_TYPE, NNZ_TYPE, ROW_TYPE, COLUMN_TYPE, BP_DATA_TYPE>(d_csr_mat_desc, &info);
  // todo: handle FLAT_PRE_CALC_BP_KERNEL_VERSION
  pre_timer.stop();

  calc_timer.start();
  using SCALE_DATA_TYPE = DATA_TYPE;
  using A_TYPE = DATA_TYPE;
  using X_TYPE = DATA_TYPE;

  using SHARE_MEM_INDEX_TYPE = INDEX_TYPE;
  using COMPUTE_MULTI_TYPE = DATA_TYPE;
  using COMPUTE_SUM_TYPE = DATA_TYPE;

  adaptive_flat_sparse_spmv<DATA_TYPE, BP_DATA_TYPE, A_TYPE, X_TYPE, Y_TYPE, NNZ_TYPE, ROW_TYPE, COLUMN_TYPE,
                            SCALE_DATA_TYPE>(hipStreamDefault, trans, alpha, beta, d_csr_mat_desc, x, y, &info);
  calc_timer.stop(true);
  if (bmt != nullptr) {
    bmt->set_time(pre_timer.time_use, calc_timer.time_use, 0.0, 0.);
  }
  return;
}

template <typename INDEX_TYPE, typename DATA_TYPE>
void segment_sum_flat_sparse_spmv(int trans, const DATA_TYPE alpha, const DATA_TYPE beta,
                                  const csr_desc<INDEX_TYPE, DATA_TYPE> h_csr_desc,
                                  const csr_desc<INDEX_TYPE, DATA_TYPE> d_csr_desc, const DATA_TYPE *x, DATA_TYPE *y,
                                  BenchmarkTime *bmt) {
  VAR_FROM_CSR_DESC(DATA_TYPE, d_csr_desc)
  const int nnz = d_csr_desc.nnz;
  const int n = d_csr_desc.cols;

  constexpr int R = 2;
  constexpr int THREADS_PER_BLOCK = 512;
  constexpr int blocks = 512;
  constexpr int RED_OPT = FLAT_REDUCE_OPTION_SEGMENT_SUM;
  flat_multi_pass_sparse_spmv<INDEX_TYPE, DATA_TYPE, R, RED_OPT, 2, blocks, THREADS_PER_BLOCK>(
      trans, alpha, beta, m, n, nnz, rowptr, colindex, value, x, y, bmt);
}

template void one_pass_flat_sparse_spmv<int, double, FLAT_PRE_CALC_BP_KERNEL_VERSION_V1>(
    int, const double, const double, const csr_desc<int, double>, const csr_desc<int, double>, const double *, double *,
    BenchmarkTime *);

template void one_pass_flat_sparse_spmv<int, double, FLAT_PRE_CALC_BP_KERNEL_VERSION_V2>(
    int, const double, const double, const csr_desc<int, double>, const csr_desc<int, double>, const double *, double *,
    BenchmarkTime *);

template void one_pass_flat_sparse_spmv<int, float, FLAT_PRE_CALC_BP_KERNEL_VERSION_V1>(int, const float, const float,
                                                                                        const csr_desc<int, float>,
                                                                                        const csr_desc<int, float>,
                                                                                        const float *, float *,
                                                                                        BenchmarkTime *);

template void one_pass_flat_sparse_spmv<int, float, FLAT_PRE_CALC_BP_KERNEL_VERSION_V2>(int, const float, const float,
                                                                                        const csr_desc<int, float>,
                                                                                        const csr_desc<int, float>,
                                                                                        const float *, float *,
                                                                                        BenchmarkTime *);

template void segment_sum_flat_sparse_spmv<int, double>(int trans, const double alpha, const double beta,
                                                        const csr_desc<int, double> h_csr_desc,
                                                        const csr_desc<int, double> d_csr_desc, const double *x,
                                                        double *y, BenchmarkTime *bmt);

template void segment_sum_flat_sparse_spmv<int, float>(int trans, const float alpha, const float beta,
                                                       const csr_desc<int, float> h_csr_desc,
                                                       const csr_desc<int, float> d_csr_desc, const float *x, float *y,
                                                       BenchmarkTime *bmt);

#endif // SPMV_ACC_BENCHMARK_SPMV_ACC_FLAT_HPP