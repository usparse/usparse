//
// Created by genshen on 2025/10/09.
//

#ifndef USPARSE_BENCHMARK_USPARSE_CSRMM_H
#define USPARSE_BENCHMARK_USPARSE_CSRMM_H

#include "csr_spmm.hpp"

template <typename I, typename T> struct UsparseCsrMMVector : CsrSpMM<I, T> {
  void csr_spmm_impl(int trans, const T alpha, const T beta, const csr_desc<I, T> h_csr_desc,
                     const csr_desc<I, T> d_csr_desc, const I M, const I K, const I N, T *B_dev, T *C_dev,
                     BenchmarkTime *bmt) override;

  bool verify_beta_c() override;

  bool trans_B_C() override;
};

template <typename I, typename T> struct RocSparseCsrMM : CsrSpMM<I, T> {
  void csr_spmm_impl(int trans, const T alpha, const T beta, const csr_desc<I, T> h_csr_desc,
                     const csr_desc<I, T> d_csr_desc, const I M, const I K, const I N, T *B_dev, T *C_dev,
                     BenchmarkTime *bmt) override;

  bool verify_beta_c() override;

  bool trans_B_C() override;
};

#endif // USPARSE_BENCHMARK_USPARSE_CSRMM_H
