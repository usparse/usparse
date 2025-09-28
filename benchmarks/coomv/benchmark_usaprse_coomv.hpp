//
// Created by genshen on 2025/9/28.
//

#ifndef USPARSE_BENCHMARK_USPARSE_COOMV
#define USPARSE_BENCHMARK_USPARSE_COOMV

#include "../../library/internal/usparse_internal.h"
#include "../utils/timer_utils.h"
#include "include/usparse.h"

struct UsparseCoomvNaiveFlat : CooSpMV {
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

struct UsparseCoomvFlat : CooSpMV {
  void coo_spmv_impl(int trans, const double alpha, const double beta, const usparse::coo_desc<int, double> h_coo_desc,
                     const usparse::coo_desc<int, double> d_coo_desc, const double *x, double *y, BenchmarkTime *bmt) {
    hip::timer::event_timer pre_timer, calc_timer, destroy_timer;

    usparse::mat::usparse_mat_info info;
    usparse::mat::usparse_create_mat_info(&info);

    pre_timer.start();
    usparse::spmv::internal::usparse_coomv_analysis_imp(usparse::usparse_operation::operation_none, d_coo_desc, &info,
                                                        usparse::spmv::usparse_coomv_algorithms::Flat);
    pre_timer.stop(true);

    calc_timer.start();
    usparse::spmv::internal::usparse_coomv_imp(usparse::usparse_operation::operation_none, alpha, beta, d_coo_desc, x,
                                               y, &info, usparse::spmv::usparse_coomv_algorithms::Flat);
    calc_timer.stop(true);

    destroy_timer.start();
    usparse::usparse_status usparse_coomv_clear_imp(usparse::mat::usparse_mat_info * info);
    destroy_timer.stop(true);

    if (bmt != nullptr) {
      bmt->set_time(pre_timer.time_use, calc_timer.time_use, 0.0, destroy_timer.time_use);
    }
  }
  bool verify_beta_y() { return true; }
};

struct UsparseCoomvLine_Enhance : CooSpMV {
  void coo_spmv_impl(int trans, const double alpha, const double beta, const usparse::coo_desc<int, double> h_coo_desc,
                     const usparse::coo_desc<int, double> d_coo_desc, const double *x, double *y, BenchmarkTime *bmt) {
    hip::timer::event_timer pre_timer, calc_timer, destroy_timer;

    usparse::mat::usparse_mat_info info;
    usparse::mat::usparse_create_mat_info(&info);

    pre_timer.start();
    usparse::spmv::internal::usparse_coomv_analysis_imp(usparse::usparse_operation::operation_none, d_coo_desc, &info,
                                                        usparse::spmv::usparse_coomv_algorithms::LineEnhance);
    pre_timer.stop(true);

    calc_timer.start();
    usparse::spmv::internal::usparse_coomv_imp(usparse::usparse_operation::operation_none, alpha, beta, d_coo_desc, x,
                                               y, &info, usparse::spmv::usparse_coomv_algorithms::LineEnhance);
    calc_timer.stop(true);

    destroy_timer.start();
    usparse::usparse_status usparse_coomv_clear_imp(usparse::mat::usparse_mat_info * info);
    destroy_timer.stop(true);

    if (bmt != nullptr) {
      bmt->set_time(pre_timer.time_use, calc_timer.time_use, 0.0, destroy_timer.time_use);
    }
  }
  bool verify_beta_y() { return true; }
};

struct UsparseCoomvVector : CooSpMV {
  void coo_spmv_impl(int trans, const double alpha, const double beta, const usparse::coo_desc<int, double> h_coo_desc,
                     const usparse::coo_desc<int, double> d_coo_desc, const double *x, double *y, BenchmarkTime *bmt) {
    hip::timer::event_timer pre_timer, calc_timer, destroy_timer;

    usparse::mat::usparse_mat_info info;
    usparse::mat::usparse_create_mat_info(&info);

    pre_timer.start();
    usparse::spmv::internal::usparse_coomv_analysis_imp(usparse::usparse_operation::operation_none, d_coo_desc, &info,
                                                        usparse::spmv::usparse_coomv_algorithms::Vector);
    pre_timer.stop(true);

    calc_timer.start();
    usparse::spmv::internal::usparse_coomv_imp(usparse::usparse_operation::operation_none, alpha, beta, d_coo_desc, x,
                                               y, &info, usparse::spmv::usparse_coomv_algorithms::Vector);
    calc_timer.stop(true);

    destroy_timer.start();
    usparse::usparse_status usparse_coomv_clear_imp(usparse::mat::usparse_mat_info * info);
    destroy_timer.stop(true);

    if (bmt != nullptr) {
      bmt->set_time(pre_timer.time_use, calc_timer.time_use, 0.0, destroy_timer.time_use);
    }
  }
  bool verify_beta_y() { return true; }
};

#endif // USPARSE_BENCHMARK_USPARSE_COOMV
