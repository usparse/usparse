#ifndef SPMV_ACC_BENCHMARK_ACSR_HPP
#define SPMV_ACC_BENCHMARK_ACSR_HPP

#define ACSR_STDERR

#include <iostream>
#include <type_traits>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "acsr/ACSR.h"
#include "utils.hpp"
#include "utils/benchmark_time.h"

template <typename I, typename T> struct ACSRSpMV : CsrSpMV<I, T> {
  void csr_spmv_impl(int trans, const T alpha, const T beta, const csr_desc<I, T> h_csr_desc,
                     const csr_desc<I, T> d_csr_desc, const T *x, T *y, BenchmarkTime *bmt) {
    acsr(trans, alpha, h_csr_desc, d_csr_desc, x, y, bmt);
  }
  bool verify_beta_y() { return false; }
};

#endif // SPMV_ACC_BENCHMARK_ACSR_SPMV
