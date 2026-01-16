//
// Created by genshen on 2025/05/08.
//

#include "usparse_internal.h"

#include "../hip/level2/coomv_kernels/coomv_flat_kernel.h"
#include "../hip/level2/coomv_kernels/coomv_spmv_acc_wrappers.h"
#include "../hip/level2/csrmv_kernels/flat/csrmv_flat.h"
#include "../hip/level2/csrmv_kernels/csr-adaptive-plus/csr_adaptive_plus_spmv.h"
#include "../hip/level2/csrmv_kernels/line-enhance/csrmv_line_enhance.h"
#include "../hip/level2/csrmv_kernels/vector-row/vector_row.h"
#include "building_config.h"

/**
 * call matrix analysis for later csrmv computation.
 */
template <typename INDEX_TYPE, typename DATA_TYPE>
usparse::usparse_status usparse::spmv::internal::usparse_csrmv_analysis_imp(
    const usparse::usparse_operation trans, const usparse::csr_desc<INDEX_TYPE, DATA_TYPE> dev_csr_mat,
    usparse::mat::usparse_mat_info *info, const usparse::spmv::usparse_csrmv_algorithms algo) {
  info->csrmv_info->flat_info = new csrmv_flat_mtx_info{};
  if (algo == usparse::spmv::usparse_csrmv_algorithms::Flat) {
    auto d_csr_mat_descr = usparse::csr_mat_descr<DATA_TYPE, INDEX_TYPE, INDEX_TYPE, INDEX_TYPE>(dev_csr_mat);
    flat_analyse<DATA_TYPE, INDEX_TYPE, INDEX_TYPE, INDEX_TYPE, INDEX_TYPE>(d_csr_mat_descr,
                                                                            info->csrmv_info->flat_info);
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
    adaptive_flat_sparse_spmv<INDEX_TYPE, DATA_TYPE, BETA_Y_POLICY_AUTO>(trans, alpha, beta, dev_csr_mat, x, y,
                                                                         info->csrmv_info->flat_info);
  } else if (algo == usparse::spmv::usparse_csrmv_algorithms::LineEnhance) {
    adaptive_enhance_sparse_spmv<DATA_TYPE, INDEX_TYPE, INDEX_TYPE, DATA_TYPE, DATA_TYPE, DATA_TYPE>(trans, alpha, beta,
                                                                                                     dev_csr_mat, x, y);
  } else if (algo == usparse::spmv::usparse_csrmv_algorithms::CsrVector) {
    vec_row_sparse_spmv<DATA_TYPE>(trans, alpha, beta, dev_csr_mat, x, y);
  }
  return usparse::usparse_status::usparse_status_success;
}

usparse::usparse_status usparse::spmv::internal::usparse_csrmv_clear_imp(usparse::mat::usparse_mat_info *info) {
  // TODO: release deviece memory and info memory.
  return usparse::usparse_status::usparse_status_not_implemented;
}

template usparse::usparse_status usparse::spmv::internal::usparse_csrmv_imp<usparse::usparse_int, usparse::usparse_fp64>(
    const usparse::usparse_operation trans, const usparse::usparse_fp64 alpha, const usparse::usparse_fp64 beta,
    const usparse::csr_desc<usparse::usparse_int, usparse::usparse_fp64> dev_csr_mat, const usparse::usparse_fp64 *x,
    usparse::usparse_fp64 *y, usparse::mat::usparse_mat_info *info, const usparse::spmv::usparse_csrmv_algorithms algo);

template usparse::usparse_status
usparse::spmv::internal::usparse_csrmv_analysis_imp<usparse::usparse_int, usparse::usparse_fp64>(
    const usparse::usparse_operation trans, const usparse::csr_desc<usparse::usparse_int, usparse::usparse_fp64> dev_csr_mat,
    usparse::mat::usparse_mat_info *info, const usparse::spmv::usparse_csrmv_algorithms algo);

template usparse::usparse_status usparse::spmv::internal::usparse_csrmv_imp<usparse::usparse_int, usparse::usparse_fp32>(
    const usparse::usparse_operation trans, const usparse::usparse_fp32 alpha, const usparse::usparse_fp32 beta,
    const usparse::csr_desc<usparse::usparse_int, usparse::usparse_fp32> dev_csr_mat, const usparse::usparse_fp32 *x,
    usparse::usparse_fp32 *y, usparse::mat::usparse_mat_info *info, const usparse::spmv::usparse_csrmv_algorithms algo);

template usparse::usparse_status
usparse::spmv::internal::usparse_csrmv_analysis_imp<usparse::usparse_int, usparse::usparse_fp32>(
    const usparse::usparse_operation trans, const usparse::csr_desc<usparse::usparse_int, usparse::usparse_fp32> dev_csr_mat,
    usparse::mat::usparse_mat_info *info, const usparse::spmv::usparse_csrmv_algorithms algo);

// implmentation of C-style api for csrmv
usparse::usparse_status
usparse_scsrmv_analysis(const usparse::usparse_operation trans,
                        const usparse::csr_desc<usparse::usparse_int, usparse::usparse_fp32> dev_csr_mat,
                        usparse::mat::usparse_mat_info *info, const usparse::spmv::usparse_csrmv_algorithms algo) {
  return usparse::spmv::internal::usparse_csrmv_analysis_imp<usparse::usparse_int, usparse::usparse_fp32>(
      trans, dev_csr_mat, info, algo);
}

usparse::usparse_status
usparse_dcsrmv_analysis(const usparse::usparse_operation trans,
                        const usparse::csr_desc<usparse::usparse_int, usparse::usparse_fp64> dev_csr_mat,
                        usparse::mat::usparse_mat_info *info, const usparse::spmv::usparse_csrmv_algorithms algo) {
  return usparse::spmv::internal::usparse_csrmv_analysis_imp<usparse::usparse_int, usparse::usparse_fp64>(
      trans, dev_csr_mat, info, algo);
}

usparse::usparse_status usparse_scsrmv(const usparse::usparse_operation trans, const usparse::usparse_fp32 alpha,
                                       const usparse::usparse_fp32 beta,
                                       const usparse::csr_desc<usparse::usparse_int, usparse::usparse_fp32> dev_csr_mat,
                                       const usparse::usparse_fp32 *x, usparse::usparse_fp32 *y,
                                       usparse::mat::usparse_mat_info *info,
                                       const usparse::spmv::usparse_csrmv_algorithms algo) {
  return usparse::spmv::internal::usparse_csrmv_imp<usparse::usparse_int, usparse::usparse_fp32>(
      trans, alpha, beta, dev_csr_mat, x, y, info, algo);
}

usparse::usparse_status usparse_dcsrmv(const usparse::usparse_operation trans, const usparse::usparse_fp64 alpha,
                                       const usparse::usparse_fp64 beta,
                                       const usparse::csr_desc<usparse::usparse_int, usparse::usparse_fp64> dev_csr_mat,
                                       const usparse::usparse_fp64 *x, usparse::usparse_fp64 *y,
                                       usparse::mat::usparse_mat_info *info,
                                       const usparse::spmv::usparse_csrmv_algorithms algo) {
  return usparse::spmv::internal::usparse_csrmv_imp<usparse::usparse_int, usparse::usparse_fp64>(
      trans, alpha, beta, dev_csr_mat, x, y, info, algo);
}

usparse::usparse_status usparse_csrmv_clear(usparse::mat::usparse_mat_info *info) {
  return usparse::spmv::internal::usparse_csrmv_clear_imp(info);
}

// For coo spmv implementation
template <typename INDEX_TYPE, typename DATA_TYPE>
usparse::usparse_status usparse::spmv::internal::usparse_coomv_analysis_imp(
    const usparse::usparse_operation trans, const usparse::coo_desc<INDEX_TYPE, DATA_TYPE> dev_coo_mat,
    usparse::mat::usparse_mat_info *info, const usparse::spmv::usparse_coomv_algorithms algo) {
  if(algo != usparse::spmv::usparse_coomv_algorithms::NaiveFlat) {
    // create matrix analysis info for coomv calculation.
    info->coomv_info = new usparse::spmv::internal::coomv_conversion_mtx_info<INDEX_TYPE>();
    usparse::spmv::internal::coomv_conv_analysis(trans, dev_coo_mat, info->coomv_info, algo);
  }
  return usparse::usparse_status::usparse_status_success;
}

template <typename INDEX_TYPE, typename DATA_TYPE>
usparse::usparse_status usparse::spmv::internal::usparse_coomv_imp(
    const usparse::usparse_operation trans, const DATA_TYPE alpha, const DATA_TYPE beta,
    const usparse::coo_desc<INDEX_TYPE, DATA_TYPE> dev_coo_mat, const DATA_TYPE *x, DATA_TYPE *y,
    usparse::mat::usparse_mat_info *info, const usparse::spmv::usparse_coomv_algorithms algo) {
  if(algo == usparse::spmv::usparse_coomv_algorithms::NaiveFlat) {
    adaptive_flat_coo_sparse_spmv<INDEX_TYPE, DATA_TYPE, BETA_Y_POLICY_AUTO>(trans, alpha, beta, dev_coo_mat, x, y);
  } else {
    usparse::spmv::internal::coomv_conv_calculate(trans, alpha, beta, dev_coo_mat, x, y, info->coomv_info, algo);
  }
  return usparse::usparse_status::usparse_status_success;
}

usparse::usparse_status usparse::spmv::internal::usparse_coomv_clear_imp(usparse::mat::usparse_mat_info *info) {
  if(info != nullptr) { // for algorithms that are not NaiveFlat
    usparse::spmv::internal::coomv_conv_clear<usparse::usparse_int>(info->coomv_info);
    delete info->coomv_info;
  }
  return usparse::usparse_status::usparse_status_success;
}

template usparse::usparse_status
usparse::spmv::internal::usparse_coomv_analysis_imp<usparse::usparse_int, usparse::usparse_fp64>(
    const usparse::usparse_operation trans,
    const usparse::coo_desc<usparse::usparse_int, usparse::usparse_fp64> dev_coo_mat,
    usparse::mat::usparse_mat_info *info, const usparse::spmv::usparse_coomv_algorithms algo);

template usparse::usparse_status
usparse::spmv::internal::usparse_coomv_analysis_imp<usparse::usparse_int, usparse::usparse_fp32>(
    const usparse::usparse_operation trans,
    const usparse::coo_desc<usparse::usparse_int, usparse::usparse_fp32> dev_coo_mat,
    usparse::mat::usparse_mat_info *info, const usparse::spmv::usparse_coomv_algorithms algo);

template usparse::usparse_status usparse::spmv::internal::usparse_coomv_imp(
    const usparse::usparse_operation trans, const usparse::usparse_fp64 alpha, const usparse::usparse_fp64 beta,
    const usparse::coo_desc<usparse::usparse_int, usparse::usparse_fp64> dev_coo_mat, const usparse::usparse_fp64 *x,
    usparse::usparse_fp64 *y, usparse::mat::usparse_mat_info *info, const usparse::spmv::usparse_coomv_algorithms algo);

template usparse::usparse_status usparse::spmv::internal::usparse_coomv_imp(
    const usparse::usparse_operation trans, const usparse::usparse_fp32 alpha, const usparse::usparse_fp32 beta,
    const usparse::coo_desc<usparse::usparse_int, usparse::usparse_fp32> dev_coo_mat, const usparse::usparse_fp32 *x,
    usparse::usparse_fp32 *y, usparse::mat::usparse_mat_info *info, const usparse::spmv::usparse_coomv_algorithms algo);
