//
// Created by genshen on 2021/11/26.
// this file is copied and modified from level2/csrmv_kernels/flat/csrmv_flat.cpp for adding timer code.
//

#ifndef SPMV_ACC_BENCHMARK_SPMV_ACC_FLAT_H
#define SPMV_ACC_BENCHMARK_SPMV_ACC_FLAT_H

#include <usparse_matrix.h>

#include "../utils/benchmark_time.h"
#include "benchmark_config.h"

template <typename INDEX_TYPE, typename DATA_TYPE,
          int FLAT_PRE_CALC_BP_KERNEL_VERSION = FLAT_PRE_CALC_BP_KERNEL_VERSION_V1>
void one_pass_flat_sparse_spmv(int trans, const DATA_TYPE alpha, const DATA_TYPE beta,
                               const csr_desc<INDEX_TYPE, DATA_TYPE> h_csr_desc,
                               const csr_desc<INDEX_TYPE, DATA_TYPE> d_csr_desc, const DATA_TYPE *x, DATA_TYPE *y,
                               BenchmarkTime *bmt);

template <typename INDEX_TYPE, typename DATA_TYPE>
void segment_sum_flat_sparse_spmv(int trans, const DATA_TYPE alpha, const DATA_TYPE beta,
                                  const csr_desc<INDEX_TYPE, DATA_TYPE> h_csr_desc,
                                  const csr_desc<INDEX_TYPE, DATA_TYPE> d_csr_desc, const DATA_TYPE *x, DATA_TYPE *y,
                                  BenchmarkTime *bmt);

template <typename INDEX_TYPE, typename DATA_TYPE>
void segment_sum_v2_flat_sparse_spmv(int trans, const DATA_TYPE alpha, const DATA_TYPE beta,
                                     const csr_desc<INDEX_TYPE, DATA_TYPE> h_csr_desc,
                                     const csr_desc<INDEX_TYPE, DATA_TYPE> d_csr_desc, const DATA_TYPE *x, DATA_TYPE *y,
                                     BenchmarkTime *bmt);

#endif // SPMV_ACC_BENCHMARK_SPMV_ACC_FLAT_HPP