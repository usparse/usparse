//
// Created by genshen on 2025/10/19.
//

#include "coo_spmm.hpp"

template <typename I, typename T> struct UsparseCooMMVector : CooSpMM<I, T> {
  void coo_spmm_impl(int trans, const T alpha, const T beta, const usparse::coo_desc<I, T> h_coo_desc,
                     const usparse::coo_desc<I, T> d_coo_desc, const I M, const I K, const I N, T *B_dev, T *C_dev,
                     BenchmarkTime *bmt) override;

  bool verify_beta_c() override;

  bool trans_B_C() override;
};
