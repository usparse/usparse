//
// Created by genshen on 2025/10/09.
//

#include "benchmark_csrmm.h"

#include "../../library/internal/usparse_internal.h"
#include "../utils/timer_utils.h"
#include "include/usparse.h"

void UsparseCsrMMVector::csr_spmm_impl(int trans, const dtype alpha, const dtype beta,
                                       const csr_desc<int, dtype> h_csr_desc, const csr_desc<int, dtype> d_csr_desc,
                                       const int M, const int K, const int N, dtype *B_dev, dtype *C_dev,
                                       BenchmarkTime *bmt) {
  usparse::usparse_handle handle;
  hip::timer::event_timer calc_timer;
  calc_timer.start();
  usparse::spmm::internal::usparse_csrmm_imp<int, dtype>(
      handle, usparse::usparse_operation::operation_none, usparse::usparse_operation::operation_none, alpha, beta,
      d_csr_desc, B_dev, N, N, C_dev, N, usparse::spmm::usparse_csrmm_algorithms::Vector);
  calc_timer.stop(true);
  double calc_time_cost = calc_timer.time_use;
  if (bmt != nullptr) {
    bmt->set_time(0., calc_time_cost, 0.0, 0.);
  }
}

bool UsparseCsrMMVector::verify_beta_c() { return true; }

bool UsparseCsrMMVector::trans_B_C() { return false; }
