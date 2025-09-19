//
// Created by genshen on 2025/05/08.
//

#include "usparse_internal.h"

#include "../hip/level2/spmv-acc-csrmv/hip-flat/spmv_hip_acc_imp.h"
#include "../hip/level2/spmv-acc-csrmv/hip-csr-adaptive-plus/csr_adaptive_plus_spmv.h"
#include "../hip/level2/spmv-acc-csrmv/hip-line-enhance/line_enhance_spmv.h"
#include "../hip/level2/spmv-acc-csrmv/hip-vector-row/vector_row.h"

/**
 * call matrix analysis for later csrmv computation.
 */
template <typename INDEX_TYPE, typename DATA_TYPE>
usparse::usparse_status usparse::spmv::internal::usparse_csrmv_analysis_imp(
    const usparse::usparse_operation trans, const usparse::csr_desc<INDEX_TYPE, DATA_TYPE> dev_csr_mat,
    usparse::mat::usparse_mat_info *info, const usparse::spmv::usparse_csrmv_algorithms algo) {
  info->csrmv_info->flat_info = new csrmv_flat_mtx_info{};
  if (algo == usparse::spmv::usparse_csrmv_algorithms::Flat) {
    flat_analyse(dev_csr_mat, info->csrmv_info->flat_info);
  } else {
    // other algorithm does not need matrix anakysis.
  }
  return usparse::usparse_status::usparse_status_success;
}

template <typename INDEX_TYPE, typename DATA_TYPE>
usparse::usparse_status usparse::spmv::internal::usparse_csrmv_imp(
    const usparse::usparse_operation trans, const DATA_TYPE alpha, const DATA_TYPE beta,
    const usparse::csr_desc<INDEX_TYPE, DATA_TYPE> dev_csr_mat, const DATA_TYPE *x, DATA_TYPE *y,
    usparse::mat::usparse_mat_info *info, const usparse::spmv::usparse_csrmv_algorithms algo) {
  if (algo == usparse::spmv::usparse_csrmv_algorithms::Flat) {
    adaptive_flat_sparse_spmv(trans, alpha, beta, dev_csr_mat, x, y, info->csrmv_info->flat_info);
  } else if (algo == usparse::spmv::usparse_csrmv_algorithms::LineEnhance) {
    adaptive_enhance_sparse_spmv(trans, alpha, beta, dev_csr_mat, x, y);
  } else if (algo == usparse::spmv::usparse_csrmv_algorithms::CsrVector) {
    vec_row_sparse_spmv(trans, alpha, beta, dev_csr_mat, x, y);
  }
  return usparse::usparse_status::usparse_status_success;
}

usparse::usparse_status usparse::spmv::internal::usparse_csrmv_clear_imp(usparse::mat::usparse_mat_info *info) {
  // TODO: release deviece memory and info memory.
  return usparse::usparse_status::usparse_status_not_implemented;
}

template usparse::usparse_status usparse::spmv::internal::usparse_csrmv_imp<int, double>(
    const usparse::usparse_operation trans, const double alpha, const double beta,
    const usparse::csr_desc<int, double> dev_csr_mat, const double *x, double *y, usparse::mat::usparse_mat_info *info,
    const usparse::spmv::usparse_csrmv_algorithms algo);

template usparse::usparse_status usparse::spmv::internal::usparse_csrmv_analysis_imp<int, double>(
    const usparse::usparse_operation trans, const usparse::csr_desc<int, double> dev_csr_mat,
    usparse::mat::usparse_mat_info *info, const usparse::spmv::usparse_csrmv_algorithms algo);
