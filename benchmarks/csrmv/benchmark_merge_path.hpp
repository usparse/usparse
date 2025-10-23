#ifndef SPMV_ACC_BENCHMARK_MERGE_PATH_HPP
#define SPMV_ACC_BENCHMARK_MERGE_PATH_HPP

#include <iostream>
#include <type_traits>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "../utils/benchmark_time.h"
#include "csr_spmv.hpp"
#include "merge-path/merge_path_config.h"
#include "merge-path/merge_path_spmv.h"
#include "utils.hpp"

template <typename I, typename T> struct MergePathSpMV : CsrSpMV<I, T> {
  void csr_spmv_impl(int trans, const T alpha, const T beta, const csr_desc<I, T> h_csr_desc,
                     const csr_desc<I, T> d_csr_desc, const T *x, T *y, BenchmarkTime *bmt) {
    merge_path_spmv(trans, alpha, beta, h_csr_desc, d_csr_desc, x, y, bmt);
  }
  bool verify_beta_y() { return false; }
};

template <typename I, typename T> struct MergePathSingleBlockUpdateSpMV : CsrSpMV<I, T> {
  void csr_spmv_impl(int trans, const T alpha, const T beta, const csr_desc<I, T> h_csr_desc,
                     const csr_desc<I, T> d_csr_desc, const T *x, T *y, BenchmarkTime *bmt) {
    merge_path_spmv<Binary, SingleBlock>(trans, alpha, beta, h_csr_desc, d_csr_desc, x, y, bmt);
  }
  bool verify_beta_y() { return false; }
};

template <typename I, typename T> struct MergePathLookBackUpdateSpMV : CsrSpMV<I, T> {
  void csr_spmv_impl(int trans, const T alpha, const T beta, const csr_desc<I, T> h_csr_desc,
                     const csr_desc<I, T> d_csr_desc, const T *x, T *y, BenchmarkTime *bmt) {
    merge_path_spmv<Binary, LookBack>(trans, alpha, beta, h_csr_desc, d_csr_desc, x, y, bmt);
  }
  bool verify_beta_y() { return false; }
};

#endif // SPMV_ACC_BENCHMARK_MERGE_PATH_HPP
