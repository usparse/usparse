//
// Created by genshen on 2025/10/08.
//

#include "usparse_internal.h"

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
