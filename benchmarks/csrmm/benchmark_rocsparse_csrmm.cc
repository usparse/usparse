//
// Created by genshen on 2025/10/09.
//

#include <hip/hip_runtime_api.h>
#include <iostream>
#include <rocsparse/rocsparse.h>

#include "../utils/timer_utils.h"
#include "benchmark_csrmm.h"
#include "csr_spmm.hpp"

#define ROCSPARSE_CHECK(stat)                                                                                          \
  {                                                                                                                    \
    if (stat != rocsparse_status_success) {                                                                            \
      std::cerr << "Error: rocsparse error in line " << __LINE__ << std::endl;                                         \
    }                                                                                                                  \
  }

// https://github.com/ROCm/rocSPARSE/blob/develop_deprecated/clients/samples/example_spmm_csr.cpp
template <typename I, typename T>
void RocSparseCsrMM<I, T>::csr_spmm_impl(int trans, const T alpha, const T beta, const csr_desc<I, T> h_csr_desc,
                                         const csr_desc<I, T> d_csr_desc, const I M, const I K, const I N, T *B_dev,
                                         T *C_dev, BenchmarkTime *bmt) {
  hip::timer::event_timer calc_timer;

  rocsparse_handle handle;
  rocsparse_create_handle(&handle);

  rocsparse_mat_descr descr;
  ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr));

  calc_timer.start();
  if constexpr (std::is_same<T, float>::value) {
    ROCSPARSE_CHECK(rocsparse_scsrmm(handle, rocsparse_operation_none, rocsparse_operation_none, M, N, K,
                                     d_csr_desc.nnz, &alpha, descr, d_csr_desc.values, d_csr_desc.row_ptr,
                                     d_csr_desc.col_index, B_dev, K, &beta, C_dev, M)); // B and C are column-order
  } else if (std::is_same<T, double>::value) {
    ROCSPARSE_CHECK(rocsparse_dcsrmm(handle, rocsparse_operation_none, rocsparse_operation_none, M, N, K,
                                     d_csr_desc.nnz, &alpha, descr, d_csr_desc.values, d_csr_desc.row_ptr,
                                     d_csr_desc.col_index, B_dev, K, &beta, C_dev, M)); // B and C are column-order
  }

  calc_timer.stop(true);

  double calc_time_cost = calc_timer.time_use;
  if (bmt != nullptr) {
    bmt->set_time(0., calc_time_cost, 0.0, 0.);
  }
}

template <typename I, typename T> bool RocSparseCsrMM<I, T>::verify_beta_c() { return true; }

template <typename I, typename T> bool RocSparseCsrMM<I, T>::trans_B_C() { return true; }

template class RocSparseCsrMM<int, float>;

template class RocSparseCsrMM<int, double>;

#undef ROCSPARSE_CHECK
