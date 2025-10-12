//
// Created by genshen on 2025/10/09.
//

#ifndef USPARSE_BENCHMARK_USPARSE_CSRMM_H
#define USPARSE_BENCHMARK_USPARSE_CSRMM_H

#include "csr_spmm.hpp"

struct UsparseCsrMMVector : CsrSpMM {
  void csr_spmm_impl(int trans, const dtype alpha, const dtype beta, const csr_desc<int, dtype> h_csr_desc,
                     const csr_desc<int, dtype> d_csr_desc, const int M, const int K, const int N, dtype *B_dev,
                     dtype *C_dev, BenchmarkTime *bmt) override;

  bool verify_beta_c() override;

  bool trans_B_C() override;
};

struct RocSparseCsrMM : CsrSpMM {
  void csr_spmm_impl(int trans, const dtype alpha, const dtype beta, const csr_desc<int, dtype> h_csr_desc,
                     const csr_desc<int, dtype> d_csr_desc, const int M, const int K, const int N, dtype *B_dev,
                     dtype *C_dev, BenchmarkTime *bmt) override;

  bool verify_beta_c() override;

  bool trans_B_C() override;
};

#endif // USPARSE_BENCHMARK_USPARSE_CSRMM_H
