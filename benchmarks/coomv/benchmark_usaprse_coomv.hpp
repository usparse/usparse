//
// Created by genshen on 2025/9/28.
//

#ifndef USPARSE_BENCHMARK_USPARSE_COOMV
#define USPARSE_BENCHMARK_USPARSE_COOMV

#include "../../library/internal/usparse_internal.h"
#include "../utils/timer_utils.h"

struct UsparseCooMVFlat : CooSpMV {
  void coo_spmv_impl(int trans, const double alpha, const double beta, const usparse::coo_desc<int, double> h_coo_desc,
                     const usparse::coo_desc<int, double> d_coo_desc, const double *x, double *y, BenchmarkTime *bmt) {
    hip::timer::event_timer calc_timer;
    calc_timer.start();
    usparse::spmv::internal::usparse_coomv_imp(usparse::usparse_operation::operation_none, alpha, beta, d_coo_desc, x,
                                               y, nullptr, usparse::spmv::usparse_coomv_algorithms::NaiveFlat);
    calc_timer.stop(true);
    double calc_time_cost = calc_timer.time_use;
    if (bmt != nullptr) {
      bmt->set_time(0., calc_time_cost, 0.0, 0.);
    }
  }
  bool verify_beta_y() { return true; }
};

#endif // USPARSE_BENCHMARK_USPARSE_COOMV
