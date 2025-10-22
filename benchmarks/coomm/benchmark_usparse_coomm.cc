//
// Created by genshen on 2025/10/19.
//

#include "benchmark_coomm.h"

#include "../../library/internal/usparse_internal.h"
#include "../utils/timer_utils.h"
#include "include/usparse.h"

template <typename I, typename T>
void UsparseCooMMVector<I, T>::coo_spmm_impl(int trans, const T alpha, const T beta,
                                             const usparse::coo_desc<I, T> h_coo_desc,
                                             const usparse::coo_desc<I, T> d_coo_desc, const I M, const I K, const I N,
                                             T *B_dev, T *C_dev, BenchmarkTime *bmt) {
  hip::timer::event_timer pre_time, calc_timer;

  usparse::usparse_handle handle;
  usparse::mat::usparse_mat_info info;
  usparse::mat::usparse_create_mat_info(&info);

  pre_time.start();
  usparse::spmm::internal::usparse_coomm_analysis_imp<I, T>(
      handle, usparse::usparse_operation::operation_none, usparse::usparse_operation::operation_none, alpha, beta,
      d_coo_desc, B_dev, N, N, C_dev, N, &info, usparse::spmm::usparse_coomm_algorithms::Vector);
  pre_time.stop(true);

  calc_timer.start();
  usparse::spmm::internal::usparse_coomm_compute_imp<I, T>(
      handle, usparse::usparse_operation::operation_none, usparse::usparse_operation::operation_none, alpha, beta,
      d_coo_desc, B_dev, N, N, C_dev, N, &info, usparse::spmm::usparse_coomm_algorithms::Vector);
  calc_timer.stop(true);
  double calc_time_cost = calc_timer.time_use;
  if (bmt != nullptr) {
    bmt->set_time(0., calc_time_cost, 0.0, 0.);
  }
}

template <typename I, typename T> bool UsparseCooMMVector<I, T>::verify_beta_c() { return true; }

template <typename I, typename T> bool UsparseCooMMVector<I, T>::trans_B_C() { return false; }

template class UsparseCooMMVector<usparse::usparse_int, usparse::usparse_fp32>;

template class UsparseCooMMVector<usparse::usparse_int, usparse::usparse_fp64>;
