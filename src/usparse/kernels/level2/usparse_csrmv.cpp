//
// Created by genshen to 2025-07-09.
//

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <rocsparse/rocsparse.h>

#include "../spmv/hip-flat/spmv_hip_acc_imp.h"
#include "../spmv/hip-line-enhance/line_enhance_spmv.h"
#include "../spmv/hip-vector-row/vector_row.h"

#include "usparse_csrmv.h"

template <typename INDEX_TYPE, typename VALUE_TYPE>
void usparse::device_csrmv(const INDEX_TYPE nrows, const INDEX_TYPE ncols, const INDEX_TYPE nnz,
                           const INDEX_TYPE *csr_row_ptr, const INDEX_TYPE *csr_col_index, const VALUE_TYPE *csr_values,
                           const VALUE_TYPE *dev_x, VALUE_TYPE *dev_y, rocsparse_handle *handle,
                           rocsparse_mat_descr *descrA, rocsparse_mat_info *info) {
  VALUE_TYPE hip_alpha = 1.0;
  VALUE_TYPE hip_beta = 0.0;
  rocsparse_dcsrmv(*handle, rocsparse_operation_none, nrows, ncols, nnz, &hip_alpha, *descrA, csr_values,
                   (rocsparse_int *)(csr_row_ptr), (rocsparse_int *)(csr_col_index), *info, dev_x, &hip_beta, dev_y);
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void usparse::device_scsrmv(const INDEX_TYPE nrows, const INDEX_TYPE ncols, const INDEX_TYPE nnz,
                            const INDEX_TYPE *csr_row_ptr, const INDEX_TYPE *csr_col_index,
                            const VALUE_TYPE *csr_values, const VALUE_TYPE *dev_x, VALUE_TYPE *dev_y,
                            rocsparse_handle *handle, rocsparse_mat_descr *descrA, rocsparse_mat_info *info) {
  VALUE_TYPE hip_alpha = 1.0f;
  VALUE_TYPE hip_beta = 0.0f;
  rocsparse_scsrmv(*handle, rocsparse_operation_none, nrows, ncols, nnz, &hip_alpha, *descrA, csr_values,
                   (rocsparse_int *)(csr_row_ptr), (rocsparse_int *)(csr_col_index), *info, dev_x, &hip_beta, dev_y);
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void usparse::device_adaptive_analysis(int trans, const VALUE_TYPE alpha, const VALUE_TYPE beta,
                                       const csr_desc<INDEX_TYPE, VALUE_TYPE> d_csr_desc,
                                       const usparse::usparse_csrmv_algo algo, usparse::csrmv_mat_info *info) {
  if (algo == usparse::usparse_csrmv_algo_flat) {
    flat_analyse(d_csr_desc, &(info->flat_info));
  } else if (algo == usparse::usparse_csrmv_algo_line_enhance) {
    return;
  } else if (algo == usparse::usparse_csrmv_algo_vector) {
    return;
  }
}

template <typename INDEX_TYPE, typename VALUE_TYPE, int BEAT_Y_POLICY>
void usparse::device_adaptive_spmv(int trans, const VALUE_TYPE alpha, const VALUE_TYPE beta,
                                   const csr_desc<INDEX_TYPE, VALUE_TYPE> d_csr_desc, const VALUE_TYPE *dx,
                                   VALUE_TYPE *dy, const usparse::usparse_csrmv_algo algo,
                                   usparse::csrmv_mat_info *info) {
  if (algo == usparse::usparse_csrmv_algo_flat && d_csr_desc.nnz > 0) {
    adaptive_flat_sparse_spmv<VALUE_TYPE, BEAT_Y_POLICY>(trans, alpha, beta, d_csr_desc, dx, dy, &(info->flat_info));
  } else if (algo == usparse::usparse_csrmv_algo_line_enhance) {
    adaptive_enhance_sparse_spmv<VALUE_TYPE, BEAT_Y_POLICY>(trans, alpha, beta, d_csr_desc, dx, dy);
  } else if (algo == usparse::usparse_csrmv_algo_vector) {
    vec_row_sparse_spmv(trans, alpha, beta, d_csr_desc, dx, dy);
  }
}

// todo: the parameter passed to spmv kernel api must int. there is a conversion from uint to int.
template void usparse::device_csrmv<uint, double>(const uint nrows, const uint ncols, const uint nnz,
                                                  const uint *csr_row_ptr, const uint *csr_col_index,
                                                  const double *csr_values, const double *dev_x, double *dev_y,
                                                  rocsparse_handle *handle, rocsparse_mat_descr *descrA,
                                                  rocsparse_mat_info *info);

// use rocsparse_scsrmv() to support float precision
template void usparse::device_scsrmv<uint, float>(const uint nrows, const uint ncols, const uint nnz,
                                                  const uint *csr_row_ptr, const uint *csr_col_index,
                                                  const float *csr_values, const float *dev_x, float *dev_y,
                                                  rocsparse_handle *handle, rocsparse_mat_descr *descrA,
                                                  rocsparse_mat_info *info);

template void usparse::device_adaptive_analysis<int, double>(int trans, const double alpha, const double beta,
                                                             const csr_desc<int, double> d_csr_desc,
                                                             const usparse::usparse_csrmv_algo algo,
                                                             usparse::csrmv_mat_info *info);

template void usparse::device_adaptive_spmv<int, double, BEAT_Y_POLICY_AUTO>(
    int trans, const double alpha, const double beta, const csr_desc<int, double> d_csr_desc, const double *dx,
    double *dy, const usparse::usparse_csrmv_algo algo, usparse::csrmv_mat_info *info);

template void usparse::device_adaptive_spmv<int, double, BEAT_Y_POLICY_ZERO>(
    int trans, const double alpha, const double beta, const csr_desc<int, double> d_csr_desc, const double *dx,
    double *dy, const usparse::usparse_csrmv_algo algo, usparse::csrmv_mat_info *info);

template void usparse::device_adaptive_spmv<int, double, BEAT_Y_POLICY_ONE>(
    int trans, const double alpha, const double beta, const csr_desc<int, double> d_csr_desc, const double *dx,
    double *dy, const usparse::usparse_csrmv_algo algo, usparse::csrmv_mat_info *info);

template void usparse::device_adaptive_analysis<int, float>(int trans, const float alpha, const float beta,
                                                            const csr_desc<int, float> d_csr_desc,
                                                            const usparse::usparse_csrmv_algo algo,
                                                            usparse::csrmv_mat_info *info);

template void usparse::device_adaptive_spmv<int, float, BEAT_Y_POLICY_AUTO>(
    int trans, const float alpha, const float beta, const csr_desc<int, float> d_csr_desc, const float *dx, float *dy,
    const usparse::usparse_csrmv_algo algo, usparse::csrmv_mat_info *info);

template void usparse::device_adaptive_spmv<int, float, BEAT_Y_POLICY_ZERO>(
    int trans, const float alpha, const float beta, const csr_desc<int, float> d_csr_desc, const float *dx, float *dy,
    const usparse::usparse_csrmv_algo algo, usparse::csrmv_mat_info *info);
template void usparse::device_adaptive_spmv<int, float, BEAT_Y_POLICY_ONE>(
    int trans, const float alpha, const float beta, const csr_desc<int, float> d_csr_desc, const float *dx, float *dy,
    const usparse::usparse_csrmv_algo algo, usparse::csrmv_mat_info *info);
