//
// Created by genshen on 2025/10/08.
//

#include "usparse_internal.h"

#include "../hip/level3/coomm_kernels/coomm_conversion_based.h"
#include "../hip/level3/csrmm_kernels/csrmm_kernel_rowsplit.h"
#include "building_config.h"
#include "include/usparse.h"

template <typename INDEX_TYPE, typename DATA_TYPE>
usparse::usparse_status usparse::spmm::internal::usparse_csrmm_imp(
    const usparse::usparse_handle handle, const usparse::usparse_operation A_trans,
    const usparse::usparse_operation B_trans, const DATA_TYPE alpha, const DATA_TYPE beta,
    const usparse::csr_desc<INDEX_TYPE, DATA_TYPE> dev_csr_mat_a, const DATA_TYPE *B_dense_val,
    const usparse::usparse_int B_cols, const INDEX_TYPE ldb, DATA_TYPE *C_dense_val, const INDEX_TYPE ldc,
    const usparse::spmm::usparse_csrmm_algorithms algo) {

  usparse::internal::adaptive_csrmm_vetor_row<INDEX_TYPE, DATA_TYPE, __WF_SIZE__>(
      A_trans, B_trans, alpha, beta, dev_csr_mat_a, B_cols, B_dense_val, C_dense_val);
  return usparse::usparse_status::usparse_status_success;
}

template <typename INDEX_TYPE, typename DATA_TYPE>
usparse::usparse_status usparse::spmm::internal::usparse_coomm_analysis_imp(
    const usparse::usparse_handle handle, const usparse::usparse_operation A_trans,
    const usparse::usparse_operation B_trans, const DATA_TYPE alpha, const DATA_TYPE beta,
    const usparse::coo_desc<INDEX_TYPE, DATA_TYPE> dev_coo_mat_a, const DATA_TYPE *B_dense_val,
    const usparse::usparse_int B_cols, const INDEX_TYPE ldb, DATA_TYPE *C_dense_val, const INDEX_TYPE ldc,
    usparse::mat::usparse_mat_info *info, const usparse::spmm::usparse_coomm_algorithms algo) {
  if (algo == usparse::spmm::usparse_coomm_algorithms::Vector) {
    // create matrix analysis info for coomv calculation.
    // todo: release memory
    info->coomm_info = new usparse::spmm::internal::coomm_conversion_mtx_info<INDEX_TYPE>();
    usparse::spmm::internal::coomm_conv_analysis(A_trans, dev_coo_mat_a, info->coomm_info, algo);
  }
  return usparse::usparse_status::usparse_status_success;
}

template <typename INDEX_TYPE, typename DATA_TYPE>
usparse::usparse_status usparse::spmm::internal::usparse_coomm_compute_imp(
    const usparse::usparse_handle handle, const usparse::usparse_operation A_trans,
    const usparse::usparse_operation B_trans, const DATA_TYPE alpha, const DATA_TYPE beta,
    const usparse::coo_desc<INDEX_TYPE, DATA_TYPE> dev_coo_mat_a, const DATA_TYPE *B_dense_val,
    const usparse::usparse_int B_cols, const INDEX_TYPE ldb, DATA_TYPE *C_dense_val, const INDEX_TYPE ldc,
    usparse::mat::usparse_mat_info *info, const usparse::spmm::usparse_coomm_algorithms algo) {
  if (algo == usparse::spmm::usparse_coomm_algorithms::Vector) {
    usparse::spmm::internal::coomm_conv_compute(A_trans, B_trans, alpha, beta, dev_coo_mat_a, B_cols, B_dense_val,
                                                C_dense_val, info->coomm_info, algo);
    return usparse::usparse_status::usparse_status_success;
  } else {
    return usparse::usparse_status::usparse_status_not_implemented;
  }
}

template usparse::usparse_status usparse::spmm::internal::usparse_csrmm_imp<int, float>(
    const usparse::usparse_handle handle, const usparse::usparse_operation A_trans,
    const usparse::usparse_operation B_trans, const float alpha, const float beta,
    const usparse::csr_desc<int, float> dev_csr_mat_a, const float *B_dense_val, const usparse::usparse_int B_cols,
    const int ldb, float *C_dense_val, const int ldc, const usparse::spmm::usparse_csrmm_algorithms algo);

template usparse::usparse_status usparse::spmm::internal::usparse_csrmm_imp<int, double>(
    const usparse::usparse_handle handle, const usparse::usparse_operation A_trans,
    const usparse::usparse_operation B_trans, const double alpha, const double beta,
    const usparse::csr_desc<int, double> dev_csr_mat_a, const double *B_dense_val, const usparse::usparse_int B_cols,
    const int ldb, double *C_dense_val, const int ldc, const usparse::spmm::usparse_csrmm_algorithms algo);

template usparse::usparse_status
usparse::spmm::internal::usparse_coomm_analysis_imp<usparse::usparse_int, usparse::usparse_fp32>(
    const usparse::usparse_handle handle, const usparse::usparse_operation A_trans,
    const usparse::usparse_operation B_trans, const usparse::usparse_fp32 alpha, const usparse::usparse_fp32 beta,
    const usparse::coo_desc<usparse::usparse_int, usparse::usparse_fp32> dev_coo_mat_a,
    const usparse::usparse_fp32 *B_dense_val, const usparse::usparse_int B_cols, const usparse::usparse_int ldb,
    usparse::usparse_fp32 *C_dense_val, const usparse::usparse_int ldc, usparse::mat::usparse_mat_info *info,
    const usparse::spmm::usparse_coomm_algorithms algo);

template usparse::usparse_status
usparse::spmm::internal::usparse_coomm_analysis_imp<usparse::usparse_int, usparse::usparse_fp64>(
    const usparse::usparse_handle handle, const usparse::usparse_operation A_trans,
    const usparse::usparse_operation B_trans, const usparse::usparse_fp64 alpha, const usparse::usparse_fp64 beta,
    const usparse::coo_desc<usparse::usparse_int, usparse::usparse_fp64> dev_coo_mat_a,
    const usparse::usparse_fp64 *B_dense_val, const usparse::usparse_int B_cols, const usparse::usparse_int ldb,
    usparse::usparse_fp64 *C_dense_val, const usparse::usparse_int ldc, usparse::mat::usparse_mat_info *info,
    const usparse::spmm::usparse_coomm_algorithms algo);

template usparse::usparse_status
usparse::spmm::internal::usparse_coomm_compute_imp<usparse::usparse_int, usparse::usparse_fp32>(
    const usparse::usparse_handle handle, const usparse::usparse_operation A_trans,
    const usparse::usparse_operation B_trans, const usparse::usparse_fp32 alpha, const usparse::usparse_fp32 beta,
    const usparse::coo_desc<usparse::usparse_int, usparse::usparse_fp32> dev_coo_mat_a,
    const usparse::usparse_fp32 *B_dense_val, const usparse::usparse_int B_cols, const usparse::usparse_int ldb,
    usparse::usparse_fp32 *C_dense_val, const usparse::usparse_int ldc, usparse::mat::usparse_mat_info *info,
    const usparse::spmm::usparse_coomm_algorithms algo);

template usparse::usparse_status
usparse::spmm::internal::usparse_coomm_compute_imp<usparse::usparse_int, usparse::usparse_fp64>(
    const usparse::usparse_handle handle, const usparse::usparse_operation A_trans,
    const usparse::usparse_operation B_trans, const usparse::usparse_fp64 alpha, const usparse::usparse_fp64 beta,
    const usparse::coo_desc<usparse::usparse_int, usparse::usparse_fp64> dev_coo_mat_a,
    const usparse::usparse_fp64 *B_dense_val, const usparse::usparse_int B_cols, const usparse::usparse_int ldb,
    usparse::usparse_fp64 *C_dense_val, const usparse::usparse_int ldc, usparse::mat::usparse_mat_info *info,
    const usparse::spmm::usparse_coomm_algorithms algo);
