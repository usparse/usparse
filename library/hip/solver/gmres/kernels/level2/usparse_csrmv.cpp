//
// Created by genshen to 2025-07-09.
//

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "level2/csrmv_kernels/flat/csrmv_flat.h"
#include "level2/csrmv_kernels/line-enhance/csrmv_line_enhance.h"
#include "level2/csrmv_kernels/vector-row/vector_row.h"
#include "level2/spmv_api_config.hpp"

#include "usparse_csrmv.h"

#ifdef __USPARSE_BUILD_BACKEND_ROCM__
#define HIPSPARSE_CHECK(stat)                                                                                          \
  {                                                                                                                    \
    if (stat != HIPSPARSE_STATUS_SUCCESS) {                                                                            \
      fprintf(stderr, "Error: hipsparse error in line %d", __LINE__);                                                  \
    }                                                                                                                  \
  }
#endif

template <typename INDEX_TYPE, typename VALUE_TYPE>
void usparse::device_adaptive_analysis(int trans, const VALUE_TYPE alpha, const VALUE_TYPE beta,
                                       const csr_desc<INDEX_TYPE, VALUE_TYPE> d_csr_desc,
                                       const usparse::usparse_csrmv_algo algo, usparse::csrmv_mat_info *info) {
  if (algo == usparse::usparse_csrmv_algo_flat) {
    auto d_csr_mat_descr = usparse::csr_mat_descr<VALUE_TYPE, INDEX_TYPE, INDEX_TYPE, INDEX_TYPE>(d_csr_desc);
    csrmv_flat_mtx_info<INDEX_TYPE, VALUE_TYPE> *flat_info =
        reinterpret_cast<csrmv_flat_mtx_info<INDEX_TYPE, VALUE_TYPE> *>(info->flat_info);
    flat_matrix_analyse_atomic<VALUE_TYPE, INDEX_TYPE, INDEX_TYPE, INDEX_TYPE, INDEX_TYPE, VALUE_TYPE>(
        hipStreamDefault, d_csr_mat_descr, flat_info);
  } else if (algo == usparse::usparse_csrmv_algo_line_enhance) {
    return;
  } else if (algo == usparse::usparse_csrmv_algo_vector) {
    return;
  } else if (algo == usparse::usparse_csrmv_algo_hipsparse_default) {
#ifdef __USPARSE_BUILD_BACKEND_ROCM__
    hipsparseHandle_t handle;
    hipsparseMatDescr_t descrA;
    hipsparseCreate(&handle);
    hipsparseCreateMatDescr(&descrA);
    info->hipsparse_default_info.handle = handle;
    info->hipsparse_default_info.descrA = descrA;
#endif
  } else if (algo == usparse::usparse_csrmv_algo_hipsparse_csr_adaptive ||
             algo == usparse::usparse_csrmv_algo_hipsparse_csr_vector) {
#ifdef __USPARSE_BUILD_BACKEND_ROCM__
    rocsparse_handle handle;
    rocsparse_create_handle(&handle);
    // Matrix descriptor
    rocsparse_mat_descr descrA;
    rocsparse_create_mat_descr(&descrA);
    rocsparse_mat_info mat_info;
    rocsparse_create_mat_info(&mat_info);

    if (algo == usparse::usparse_csrmv_algo_hipsparse_csr_adaptive) {
      // TODO: use analysis of type fp32
      rocsparse_scsrmv_analysis(handle, rocsparse_operation_none, d_csr_desc.rows, d_csr_desc.cols, d_csr_desc.nnz,
                                descrA, (float *)d_csr_desc.values, d_csr_desc.row_ptr, d_csr_desc.col_index, mat_info);
    }

    info->hipsparse_csr_adaptive_info.handle = handle;
    info->hipsparse_csr_adaptive_info.descrA = descrA;
    info->hipsparse_csr_adaptive_info.info = mat_info;
#endif
  }
}

template <typename INDEX_TYPE, typename VALUE_TYPE, int BETA_Y_POLICY>
void usparse::device_adaptive_spmv(int trans, const VALUE_TYPE alpha, const VALUE_TYPE beta,
                                   const csr_desc<INDEX_TYPE, VALUE_TYPE> d_csr_desc, const VALUE_TYPE *dx,
                                   VALUE_TYPE *dy, const usparse::usparse_csrmv_algo algo,
                                   usparse::csrmv_mat_info *info) {
  if (algo == usparse::usparse_csrmv_algo_flat && d_csr_desc.nnz > 0) {
    csrmv_flat_mtx_info<INDEX_TYPE, VALUE_TYPE> *flat_info =
        reinterpret_cast<csrmv_flat_mtx_info<INDEX_TYPE, VALUE_TYPE> *>(info->flat_info);
    adaptive_csrmv_flat_atomic<VALUE_TYPE, INDEX_TYPE, VALUE_TYPE, VALUE_TYPE, VALUE_TYPE, INDEX_TYPE, INDEX_TYPE,
                               INDEX_TYPE, VALUE_TYPE, BETA_Y_POLICY>(hipStreamDefault, trans, alpha, beta, d_csr_desc,
                                                                      dx, dy, flat_info);
  } else if (algo == usparse::usparse_csrmv_algo_line_enhance) {
    auto d_csr_mat_descr = usparse::csr_mat_descr<VALUE_TYPE, INDEX_TYPE, INDEX_TYPE, INDEX_TYPE>(d_csr_desc);
    adaptive_enhance_sparse_spmv<VALUE_TYPE, INDEX_TYPE, INDEX_TYPE, INDEX_TYPE, VALUE_TYPE, VALUE_TYPE, VALUE_TYPE,
                                 VALUE_TYPE>(trans, alpha, beta, d_csr_mat_descr, dx, dy);
  } else if (algo == usparse::usparse_csrmv_algo_vector) {
    vec_row_sparse_spmv(trans, alpha, beta, d_csr_desc, dx, dy);
  } else if (algo == usparse::usparse_csrmv_algo_hipsparse_default) {
#ifdef __USPARSE_BUILD_BACKEND_ROCM__
    const int m = d_csr_desc.rows;
    const int n = d_csr_desc.cols;
    const int nnz = d_csr_desc.nnz;

    const VALUE_TYPE *dAval = d_csr_desc.values;
    const INDEX_TYPE *dAptr = d_csr_desc.row_ptr;
    const INDEX_TYPE *dAcol = d_csr_desc.col_index;

    // 2. alpha, beta 必须传 host pointer
    VALUE_TYPE halpha = alpha;
    VALUE_TYPE hbeta = beta;

    if constexpr (std::is_same<VALUE_TYPE, double>::value) {
      HIPSPARSE_CHECK(hipsparseDcsrmv(info->hipsparse_default_info.handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, m, n, nnz,
                                      &halpha, info->hipsparse_default_info.descrA, dAval, dAptr, dAcol, dx, &hbeta,
                                      dy));
    } else if constexpr (std::is_same<VALUE_TYPE, float>::value) {
      HIPSPARSE_CHECK(hipsparseScsrmv(info->hipsparse_default_info.handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, m, n, nnz,
                                      &halpha, info->hipsparse_default_info.descrA, dAval, dAptr, dAcol, dx, &hbeta,
                                      dy));
    }
#endif
  } else if (algo == usparse::usparse_csrmv_algo_hipsparse_csr_adaptive ||
             algo == usparse::usparse_csrmv_algo_hipsparse_csr_vector) {
#ifdef __USPARSE_BUILD_BACKEND_ROCM__
    VALUE_TYPE hip_alpha = alpha;
    VALUE_TYPE hip_beta = beta;
    if constexpr (std::is_same<VALUE_TYPE, double>::value) {
      rocsparse_dcsrmv(info->hipsparse_csr_adaptive_info.handle, rocsparse_operation_none, d_csr_desc.rows,
                       d_csr_desc.cols, d_csr_desc.nnz, &hip_alpha, info->hipsparse_csr_adaptive_info.descrA,
                       d_csr_desc.values, d_csr_desc.row_ptr, d_csr_desc.col_index,
                       info->hipsparse_csr_adaptive_info.info, dx, &hip_beta, dy);
    } else if constexpr (std::is_same<VALUE_TYPE, float>::value) {
      rocsparse_scsrmv(info->hipsparse_csr_adaptive_info.handle, rocsparse_operation_none, d_csr_desc.rows,
                       d_csr_desc.cols, d_csr_desc.nnz, &hip_alpha, info->hipsparse_csr_adaptive_info.descrA,
                       d_csr_desc.values, d_csr_desc.row_ptr, d_csr_desc.col_index,
                       info->hipsparse_csr_adaptive_info.info, dx, &hip_beta, dy);
    }
#endif
  }
}

// todo: the parameter passed to spmv kernel api must int. there is a conversion from uint to int.
template void usparse::device_adaptive_analysis<int, double>(int trans, const double alpha, const double beta,
                                                             const csr_desc<int, double> d_csr_desc,
                                                             const usparse::usparse_csrmv_algo algo,
                                                             usparse::csrmv_mat_info *info);

template void usparse::device_adaptive_spmv<int, double, BETA_Y_POLICY_AUTO>(
    int trans, const double alpha, const double beta, const csr_desc<int, double> d_csr_desc, const double *dx,
    double *dy, const usparse::usparse_csrmv_algo algo, usparse::csrmv_mat_info *info);

template void usparse::device_adaptive_spmv<int, double, BETA_Y_POLICY_ZERO>(
    int trans, const double alpha, const double beta, const csr_desc<int, double> d_csr_desc, const double *dx,
    double *dy, const usparse::usparse_csrmv_algo algo, usparse::csrmv_mat_info *info);

template void usparse::device_adaptive_spmv<int, double, BETA_Y_POLICY_ONE>(
    int trans, const double alpha, const double beta, const csr_desc<int, double> d_csr_desc, const double *dx,
    double *dy, const usparse::usparse_csrmv_algo algo, usparse::csrmv_mat_info *info);

template void usparse::device_adaptive_analysis<int, float>(int trans, const float alpha, const float beta,
                                                            const csr_desc<int, float> d_csr_desc,
                                                            const usparse::usparse_csrmv_algo algo,
                                                            usparse::csrmv_mat_info *info);

template void usparse::device_adaptive_spmv<int, float, BETA_Y_POLICY_AUTO>(
    int trans, const float alpha, const float beta, const csr_desc<int, float> d_csr_desc, const float *dx, float *dy,
    const usparse::usparse_csrmv_algo algo, usparse::csrmv_mat_info *info);

template void usparse::device_adaptive_spmv<int, float, BETA_Y_POLICY_ZERO>(
    int trans, const float alpha, const float beta, const csr_desc<int, float> d_csr_desc, const float *dx, float *dy,
    const usparse::usparse_csrmv_algo algo, usparse::csrmv_mat_info *info);
template void usparse::device_adaptive_spmv<int, float, BETA_Y_POLICY_ONE>(
    int trans, const float alpha, const float beta, const csr_desc<int, float> d_csr_desc, const float *dx, float *dy,
    const usparse::usparse_csrmv_algo algo, usparse::csrmv_mat_info *info);
