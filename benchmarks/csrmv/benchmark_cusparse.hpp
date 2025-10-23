//
// Created by reget on 2021/11/16.
//

#ifndef SPMV_ACC_BENCHMARK_CUSPARSE_HPP
#define SPMV_ACC_BENCHMARK_CUSPARSE_HPP

#include <cusparse.h>

#include <usparse_matrix.h>

#include "timer/timer.h"
#include "../utils/benchmark_time.h"
#include "../utils/timer_utils.h"

template <typename I, typename T> struct CuSparseGeneral : CsrSpMV<I, T> {

  inline cusparseSpMVAlg_t get_spmv_algo() {
// CUSPARSE_MV_ALG_DEFAULT is deprecated in 11.2.1 and removed in 11.2.2
#if CUSPARSE_VERSION < 11201
    return CUSPARSE_MV_ALG_DEFAULT;
#else
    return CUSPARSE_SPMV_ALG_DEFAULT;
#endif
  }

  void csr_spmv_impl(int trans, const T alpha, const T beta, const csr_desc<I, T> h_csr_desc,
                     const csr_desc<I, T> d_csr_desc, const T *x, T *y, BenchmarkTime *bmt) {

    const T cu_alpha = static_cast<T>(alpha);
    const T cu_beta = static_cast<T>(beta);
    hip::timer::event_timer pre_timer, calc_timer, destroy_timer;

    auto ttype = CUDA_R_64F;
    auto iitype = CUSPARSE_INDEX_32I;
    if (std::is_same<I, float>::value) {
      ttype = CUDA_R_32F;
    }

    pre_timer.start();
    // Create cuSPARSE handle
    cusparseHandle_t handle = NULL;
    cusparseCreate(&handle);
    // Create matrix, vector x and vector y
    cusparseSpMatDescr_t cu_mat;
    cusparseDnVecDescr_t cu_x, cu_y;
    cusparseCreateCsr(&cu_mat, h_csr_desc.rows, h_csr_desc.cols, h_csr_desc.nnz, (void *)d_csr_desc.row_ptr,
                      (void *)d_csr_desc.col_index, (void *)d_csr_desc.values, iitype, iitype, CUSPARSE_INDEX_BASE_ZERO,
                      ttype);
    cusparseCreateDnVec(&cu_x, h_csr_desc.cols, (void *)x, ttype);
    cusparseCreateDnVec(&cu_y, h_csr_desc.rows, (void *)y, ttype);
    // Allocate an external buffer
    void *d_buffer = NULL;
    size_t buffer_size = 0;
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &cu_alpha, cu_mat, cu_x, &cu_beta, cu_y, ttype,
                            get_spmv_algo(), &buffer_size);
    cudaMalloc(&d_buffer, buffer_size);
    pre_timer.stop();
    calc_timer.start();
    // Execute SpMV
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &cu_alpha, cu_mat, cu_x, &cu_beta, cu_y, ttype,
                 get_spmv_algo(), d_buffer);
    calc_timer.stop(true);
    destroy_timer.start();
    // Clear up on device
    cusparseDestroySpMat(cu_mat);
    cusparseDestroyDnVec(cu_x);
    cusparseDestroyDnVec(cu_y);
    cusparseDestroy(handle);
    destroy_timer.stop();
    if (bmt != nullptr) {
      bmt->set_time(pre_timer.time_use, calc_timer.time_use, 0.0, destroy_timer.time_use);
    }
  }
  bool verify_beta_y() { return true; }
};

#endif // SPMV_ACC_BENCHMARK_CUSPARSE_HPP
