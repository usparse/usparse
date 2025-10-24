//
// Created by genshen on 2025/9/28.
//

#ifndef USPARSE_BENCHMARK_ROCSPARSE_COOMV
#define USPARSE_BENCHMARK_ROCSPARSE_COOMV

#include <hip/hip_runtime_api.h>
#include <rocsparse/rocsparse.h>

template <typename INDEX_TYPE, typename DATA_TYPE> struct RocSparseCooMV : CooSpMV<INDEX_TYPE, DATA_TYPE> {
  void coo_spmv_impl(int trans, const DATA_TYPE alpha, const DATA_TYPE beta,
                     const usparse::coo_desc<INDEX_TYPE, DATA_TYPE> h_coo_desc,
                     const usparse::coo_desc<INDEX_TYPE, DATA_TYPE> d_coo_desc, const DATA_TYPE *x, DATA_TYPE *y,
                     BenchmarkTime *bmt) {

    // rocSPARSE handle
    rocsparse_handle handle;
    rocsparse_create_handle(&handle);

    // Matrix descriptor
    rocsparse_mat_descr descrA;
    rocsparse_create_mat_descr(&descrA);

    hip::timer::event_timer calc_timer;
    calc_timer.start();
    if constexpr (std::is_same<DATA_TYPE, float>::value) { // float
      rocsparse_scoomv(handle, rocsparse_operation_none, d_coo_desc.rows, d_coo_desc.cols, d_coo_desc.nnz, &alpha,
                       descrA, d_coo_desc.values, d_coo_desc.row_index, d_coo_desc.col_index, x, &beta, y);
    } else if constexpr (std::is_same<DATA_TYPE, double>::value) { // double
      rocsparse_dcoomv(handle, rocsparse_operation_none, d_coo_desc.rows, d_coo_desc.cols, d_coo_desc.nnz, &alpha,
                       descrA, d_coo_desc.values, d_coo_desc.row_index, d_coo_desc.col_index, x, &beta, y);
    }
    calc_timer.stop(true);
    double calc_time_cost = calc_timer.time_use;
    if (bmt != nullptr) {
      bmt->set_time(0., calc_time_cost, 0.0, 0.);
    }
  }
  bool verify_beta_y() { return true; }
};

#endif // USPARSE_BENCHMARK_ROCSPARSE_COOMV
