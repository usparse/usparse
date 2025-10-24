//
// Created by genshen on 2025/9/28.
//

#ifndef USPARSE_BENCHMARK_USPARSE_COOMV
#define USPARSE_BENCHMARK_USPARSE_COOMV

#include "../../library/internal/usparse_internal.h"
#include "../utils/timer_utils.h"
#include "include/usparse.h"

template <typename INDEX_TYPE, typename DATA_TYPE> struct UsparseCoomvNaiveFlat : CooSpMV<INDEX_TYPE, DATA_TYPE> {
  void coo_spmv_impl(int trans, const DATA_TYPE alpha, const DATA_TYPE beta,
                     const usparse::coo_desc<INDEX_TYPE, DATA_TYPE> h_coo_desc,
                     const usparse::coo_desc<INDEX_TYPE, DATA_TYPE> d_coo_desc, const DATA_TYPE *x, DATA_TYPE *y,
                     BenchmarkTime *bmt) {
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

template <typename INDEX_TYPE, typename DATA_TYPE> struct UsparseCoomvFlat : CooSpMV<INDEX_TYPE, DATA_TYPE> {
  void coo_spmv_impl(int trans, const DATA_TYPE alpha, const DATA_TYPE beta,
                     const usparse::coo_desc<int, DATA_TYPE> h_coo_desc,
                     const usparse::coo_desc<INDEX_TYPE, DATA_TYPE> d_coo_desc, const DATA_TYPE *x, DATA_TYPE *y,
                     BenchmarkTime *bmt) {
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

template <typename INDEX_TYPE, typename DATA_TYPE> struct UsparseCoomvLine_Enhance : CooSpMV<INDEX_TYPE, DATA_TYPE> {
  void coo_spmv_impl(int trans, const DATA_TYPE alpha, const DATA_TYPE beta,
                     const usparse::coo_desc<INDEX_TYPE, DATA_TYPE> h_coo_desc,
                     const usparse::coo_desc<INDEX_TYPE, DATA_TYPE> d_coo_desc, const DATA_TYPE *x, DATA_TYPE *y,
                     BenchmarkTime *bmt) {
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

template <typename INDEX_TYPE, typename DATA_TYPE> struct UsparseCoomvVector : CooSpMV<INDEX_TYPE, DATA_TYPE> {
  void coo_spmv_impl(int trans, const DATA_TYPE alpha, const DATA_TYPE beta,
                     const usparse::coo_desc<INDEX_TYPE, DATA_TYPE> h_coo_desc,
                     const usparse::coo_desc<INDEX_TYPE, DATA_TYPE> d_coo_desc, const DATA_TYPE *x, DATA_TYPE *y,
                     BenchmarkTime *bmt) {
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
