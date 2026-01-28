//
// Created by reget on 2021/11/16.
//

#ifndef SPMV_ACC_BENCHMARK_SPMV_ACC_HPP
#define SPMV_ACC_BENCHMARK_SPMV_ACC_HPP

#include "usparse_matrix.h"
#include "../utils/timer_utils.h"
#include "../utils/benchmark_time.h"

#include "csr_spmv.hpp"

#include "api/handle.h"
#include "flat/spmv_acc_flat.h"

#include "level2/csrmv_kernels/adaptive/adaptive.h"
#include "level2/csrmv_kernels/csr-adaptive-plus/csr_adaptive_plus_spmv.h"
#include "level2/csrmv_kernels/light/spmv_hip_acc_imp.h"
#include "level2/csrmv_kernels/line-enhance/csrmv_line_enhance.h"
#include "level2/csrmv_kernels/line/line_strategy.h"
#include "level2/csrmv_kernels/thread-row/thread_row.h"
#include "level2/csrmv_kernels/vector-row/vector_row.h"

#include "../../library/internal/usparse_internal.h"

template <typename I, typename T> struct SpMVAccAdaptive : CsrSpMV<I, T> {
  void csr_spmv_impl(int trans, const T alpha, const T beta, const csr_desc<I, T> h_csr_desc,
                     const csr_desc<I, T> d_csr_desc, const T *x, T *y, BenchmarkTime *bmt) {
    hip::timer::event_timer calc_timer;
    calc_timer.start();
    adaptive_sparse_spmv(trans, alpha, beta, h_csr_desc, d_csr_desc, x, y);
    calc_timer.stop(true);
    double calc_time_cost = calc_timer.time_use;
    if (bmt != nullptr) {
      bmt->set_time(0., calc_time_cost, 0.0, 0.);
    }
  }
  bool verify_beta_y() { return true; }
};

template <typename I, typename T> struct SpMVAccFlat : CsrSpMV<I, T> {
  void csr_spmv_impl(int trans, const T alpha, const T beta, const csr_desc<I, T> h_csr_desc,
                     const csr_desc<I, T> d_csr_desc, const T *x, T *y, BenchmarkTime *bmt) {
    one_pass_flat_sparse_spmv<I, T, FLAT_PRE_CALC_BP_KERNEL_VERSION_V1>(trans, alpha, beta, h_csr_desc, d_csr_desc, x,
                                                                        y, bmt);
  }
  bool verify_beta_y() { return true; }
};

template <typename I, typename T> struct SpMVAccFlatV2 : CsrSpMV<I, T> {
  void csr_spmv_impl(int trans, const T alpha, const T beta, const csr_desc<I, T> h_csr_desc,
                     const csr_desc<I, T> d_csr_desc, const T *x, T *y, BenchmarkTime *bmt) {
    one_pass_flat_sparse_spmv<I, T, FLAT_PRE_CALC_BP_KERNEL_VERSION_V2>(trans, alpha, beta, h_csr_desc, d_csr_desc, x,
                                                                        y, bmt);
  }
  bool verify_beta_y() { return true; }
};

template <typename I, typename T> struct SpMVAccFlatSegSum : CsrSpMV<I, T> {
  void csr_spmv_impl(int trans, const T alpha, const T beta, const csr_desc<I, T> h_csr_desc,
                     const csr_desc<I, T> d_csr_desc, const T *x, T *y, BenchmarkTime *bmt) {
    segment_sum_flat_sparse_spmv<I, T>(trans, alpha, beta, h_csr_desc, d_csr_desc, x, y, bmt);
  }
  bool verify_beta_y() { return true; }
};

template <typename I, typename T> struct SpMVAccFlatSegSumV2 : CsrSpMV<I, T> {
  void csr_spmv_impl(int trans, const T alpha, const T beta, const csr_desc<I, T> h_csr_desc,
                     const csr_desc<I, T> d_csr_desc, const T *x, T *y, BenchmarkTime *bmt) {
    segment_sum_v2_flat_sparse_spmv<I, T>(trans, alpha, beta, h_csr_desc, d_csr_desc, x, y, bmt);
  }
  bool verify_beta_y() { return true; }
};

template <typename I, typename T> struct SpMVAccLight : CsrSpMV<I, T> {
  void csr_spmv_impl(int trans, const T alpha, const T beta, const csr_desc<I, T> h_csr_desc,
                     const csr_desc<I, T> d_csr_desc, const T *x, T *y, BenchmarkTime *bmt) {
    hip::timer::event_timer calc_timer;
    calc_timer.start();
    light_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
    calc_timer.stop(true);
    double calc_time_cost = calc_timer.time_use;
    if (bmt != nullptr) {
      bmt->set_time(0., calc_time_cost, 0.0, 0.);
    }
  }
  bool verify_beta_y() { return true; }
};

template <typename I, typename T> struct SpMVAccLine : CsrSpMV<I, T> {
  void csr_spmv_impl(int trans, const T alpha, const T beta, const csr_desc<I, T> h_csr_desc,
                     const csr_desc<I, T> d_csr_desc, const T *x, T *y, BenchmarkTime *bmt) {
    hip::timer::event_timer calc_timer;
    calc_timer.start();
    adaptive_line_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
    calc_timer.stop(true);
    double calc_time_cost = calc_timer.time_use;
    if (bmt != nullptr) {
      bmt->set_time(0., calc_time_cost, 0.0, 0.);
    }
  }
  bool verify_beta_y() { return true; }
};

template <typename I, typename T> struct SpMVAccThreadRow : CsrSpMV<I, T> {
  void csr_spmv_impl(int trans, const T alpha, const T beta, const csr_desc<I, T> h_csr_desc,
                     const csr_desc<I, T> d_csr_desc, const T *x, T *y, BenchmarkTime *bmt) {
    hip::timer::event_timer calc_timer;
    calc_timer.start();
    thread_row_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
    calc_timer.stop(true);
    double calc_time_cost = calc_timer.time_use;
    if (bmt != nullptr) {
      bmt->set_time(0., calc_time_cost, 0.0, 0.);
    }
  }
  bool verify_beta_y() { return true; }
};

template <typename I, typename T> struct SpMVAccVecRow : CsrSpMV<I, T> {
  void csr_spmv_impl(int trans, const T alpha, const T beta, const csr_desc<I, T> h_csr_desc,
                     const csr_desc<I, T> d_csr_desc, const T *x, T *y, BenchmarkTime *bmt) {
    hip::timer::event_timer calc_timer;
    calc_timer.start();
    usparse::spmv::internal::usparse_csrmv_imp(usparse::usparse_operation::operation_none, alpha, beta, d_csr_desc, x,
                                               y, nullptr, usparse::spmv::usparse_csrmv_algorithms::CsrVector);
    calc_timer.stop(true);
    double calc_time_cost = calc_timer.time_use;
    if (bmt != nullptr) {
      bmt->set_time(0., calc_time_cost, 0.0, 0.);
    }
  }
  bool verify_beta_y() { return true; }
};

template <typename I, typename T> struct SpMVAccLineEnhance : CsrSpMV<I, T> {
  void csr_spmv_impl(int trans, const T alpha, const T beta, const csr_desc<I, T> h_csr_desc,
                     const csr_desc<I, T> d_csr_desc, const T *x, T *y, BenchmarkTime *bmt) {
    hip::timer::event_timer calc_timer;
    calc_timer.start();
    usparse::spmv::internal::usparse_csrmv_imp(usparse::usparse_operation::operation_none, alpha, beta, d_csr_desc, x,
                                               y, nullptr, usparse::spmv::usparse_csrmv_algorithms::LineEnhance);
    calc_timer.stop(true);
    double calc_time_cost = calc_timer.time_use;
    if (bmt != nullptr) {
      bmt->set_time(0., calc_time_cost, 0.0, 0.);
    }
  }
  bool verify_beta_y() { return true; }
};

template <typename I, typename T> struct SpMVAccAdaptivePlus : CsrSpMV<I, T> {
  void csr_spmv_impl(int trans, const T alpha, const T beta, const csr_desc<I, T> h_csr_desc,
                     const csr_desc<I, T> d_csr_desc, const T *x, T *y, BenchmarkTime *bmt) {
    SpMVAccHanele handle;
    csr_adaptive_plus_sparse_spmv<true, I, T>(&handle, trans, alpha, beta, h_csr_desc, d_csr_desc, x, y);

    double pre_time_cost = handle.profile_analyze_time;
    double calc_time_cost = handle.profile_kernel_time;
    double destroy_time_cost = handle.profile_destroy_time;
    if (bmt != nullptr) {
      bmt->set_time(pre_time_cost, calc_time_cost, 0.0, destroy_time_cost);
    }
  }
  bool verify_beta_y() { return true; }
};

#endif // SPMV_ACC_BENCHMARK_SPMV_ACC_HPP
