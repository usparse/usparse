//
// Created by genshen on 2025/10/09.
//

#include "benchmark_csrmm.h"

#include "../../library/internal/usparse_internal.h"
#include "../utils/timer_utils.h"
#include "include/usparse.h"

template <typename I, typename T>
void UsparseCsrMMVector<I, T>::csr_spmm_impl(int trans, const T alpha, const T beta, const csr_desc<I, T> h_csr_desc,
                                             const csr_desc<I, T> d_csr_desc, const I M, const I K, const I N, T *B_dev,
                                             T *C_dev, BenchmarkTime *bmt) {
  usparse::usparse_handle handle;
  hip::timer::event_timer calc_timer;
  calc_timer.start();
  usparse::spmm::internal::usparse_csrmm_imp<I, T>(
      handle, usparse::usparse_operation::operation_none, usparse::usparse_operation::operation_none, alpha, beta,
      d_csr_desc, B_dev, N, N, C_dev, N, usparse::spmm::usparse_csrmm_algorithms::Vector);
  calc_timer.stop(true);
  double calc_time_cost = calc_timer.time_use;
  if (bmt != nullptr) {
    bmt->set_time(0., calc_time_cost, 0.0, 0.);
  }
}

template <typename I, typename T> bool UsparseCsrMMVector<I, T>::verify_beta_c() { return true; }

template <typename I, typename T> bool UsparseCsrMMVector<I, T>::trans_B_C() { return false; }

template class UsparseCsrMMVector<usparse::usparse_int, usparse::usparse_fp32>;

template class UsparseCsrMMVector<usparse::usparse_int, usparse::usparse_fp64>;
