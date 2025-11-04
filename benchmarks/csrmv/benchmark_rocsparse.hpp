//
// Created by reget on 2021/11/22.
//

#ifndef SPMV_ACC_BENCHMARK_ROCSPARSE_HPP
#define SPMV_ACC_BENCHMARK_ROCSPARSE_HPP

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <hipsparse.h> // cusparseSpMV
#include <rocsparse.h>

#include <type_traits>
#include <usparse_matrix.h>

#include "../utils/benchmark_time.h"
#include "timer/timer.h"

template <typename I, typename T> struct RocSparseVecRow : CsrSpMV<I, T> {
  void csr_spmv_impl(int trans, const T alpha, const T beta, const csr_desc<I, T> h_csr_desc,
                     const csr_desc<I, T> d_csr_desc, const T *x, T *y, BenchmarkTime *bmt) {
    const T hip_alpha = static_cast<T>(alpha);
    const T hip_beta = static_cast<T>(beta);
    hip::timer::event_timer pre_timer, calc_timer, destroy_timer;
    pre_timer.start();
    // rocSPARSE handle
    rocsparse_handle handle;
    rocsparse_create_handle(&handle);
    // Matrix descriptor
    rocsparse_mat_descr descrA;
    rocsparse_create_mat_descr(&descrA);
    pre_timer.stop();
    calc_timer.start();
    if constexpr (std::is_same<T, float>::value) { // float
      rocsparse_scsrmv(handle, rocsparse_operation_none, d_csr_desc.rows, d_csr_desc.cols, d_csr_desc.nnz, &hip_alpha,
                       descrA, d_csr_desc.values, d_csr_desc.row_ptr, d_csr_desc.col_index, nullptr, x, &hip_beta, y);
    } else { // double
      rocsparse_dcsrmv(handle, rocsparse_operation_none, d_csr_desc.rows, d_csr_desc.cols, d_csr_desc.nnz, &hip_alpha,
                       descrA, d_csr_desc.values, d_csr_desc.row_ptr, d_csr_desc.col_index, nullptr, x, &hip_beta, y);
    }
    calc_timer.stop(true);
    destroy_timer.start();
    // Clear up on device
    rocsparse_destroy_mat_descr(descrA);
    rocsparse_destroy_handle(handle);
    destroy_timer.stop();
    if (bmt != nullptr) {
      bmt->set_time(pre_timer.time_use, calc_timer.time_use, 0.0, destroy_timer.time_use);
    }
  }
  bool verify_beta_y() { return true; }
};

template <typename I, typename T> struct RocSparseAdaptive : CsrSpMV<I, T> {
  void csr_spmv_impl(int trans, const T alpha, const T beta, const csr_desc<I, T> h_csr_desc,
                     const csr_desc<I, T> d_csr_desc, const T *x, T *y, BenchmarkTime *bmt) {
    const T hip_alpha = static_cast<T>(alpha);
    const T hip_beta = static_cast<T>(beta);
    hip::timer::event_timer pre_timer, calc_timer, destroy_timer;
    pre_timer.start();
    // rocSPARSE handle
    rocsparse_handle handle;
    rocsparse_create_handle(&handle);
    // Matrix descriptor
    rocsparse_mat_descr descrA;
    rocsparse_create_mat_descr(&descrA);
    // Create meta data
    rocsparse_mat_info info;
    rocsparse_create_mat_info(&info);
    // Analyse CSR matrix
    if constexpr (std::is_same<T, float>::value) { // float
      rocsparse_scsrmv_analysis(handle, rocsparse_operation_none, d_csr_desc.rows, d_csr_desc.cols, d_csr_desc.nnz,
                                descrA, d_csr_desc.values, d_csr_desc.row_ptr, d_csr_desc.col_index, info);
    } else { // double
      rocsparse_dcsrmv_analysis(handle, rocsparse_operation_none, d_csr_desc.rows, d_csr_desc.cols, d_csr_desc.nnz,
                                descrA, d_csr_desc.values, d_csr_desc.row_ptr, d_csr_desc.col_index, info);
    }
    pre_timer.stop();
    calc_timer.start();
    if constexpr (std::is_same<T, float>::value) { // float
      rocsparse_scsrmv(handle, rocsparse_operation_none, d_csr_desc.rows, d_csr_desc.cols, d_csr_desc.nnz, &hip_alpha,
                       descrA, d_csr_desc.values, d_csr_desc.row_ptr, d_csr_desc.col_index, info, x, &hip_beta, y);
    } else { // double
      rocsparse_dcsrmv(handle, rocsparse_operation_none, d_csr_desc.rows, d_csr_desc.cols, d_csr_desc.nnz, &hip_alpha,
                       descrA, d_csr_desc.values, d_csr_desc.row_ptr, d_csr_desc.col_index, info, x, &hip_beta, y);
    }
    calc_timer.stop(true);
    destroy_timer.start();
    // Clear up on device
    rocsparse_destroy_mat_info(info);
    rocsparse_destroy_mat_descr(descrA);
    rocsparse_destroy_handle(handle);
    destroy_timer.stop();
    if (bmt != nullptr) {
      bmt->set_time(pre_timer.time_use, calc_timer.time_use, 0.0, destroy_timer.time_use);
    }
  }
  bool verify_beta_y() { return true; }
};

#endif // SPMV_ACC_BENCHMARK_ROCSPARSE_HPP
