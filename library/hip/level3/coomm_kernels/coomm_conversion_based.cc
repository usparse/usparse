/**
 * Created by genshen on 2025/10/19.
 *
 */

#include "include/usparse.h"

#include "../../conversion/coo2csr.h"
#include "../csrmm_kernels/csrmm_kernel_rowsplit.h"
#include "building_config.h"
#include "coomm_conversion_based.h"
#include "coomm_mtx_analysis_info.h"
#include "common/usparse_utils.hpp"

template <typename INDEX_TYPE, typename DATA_TYPE>
void usparse::spmm::internal::coomm_conv_analysis(const usparse::usparse_operation trans,
                                                  const usparse::coo_desc<INDEX_TYPE, DATA_TYPE> d_coo_desc,
                                                  usparse::spmm::internal::coomm_conversion_mtx_info<INDEX_TYPE> *info,
                                                  const usparse::spmm::usparse_coomm_algorithms algo) {
  INDEX_TYPE *d_row_ptr;
  USPARSE_HIP_API_CHECK(hipMalloc((void **)&d_row_ptr, sizeof(int) * (d_coo_desc.rows + 1)));
  usparse::conversion::coo2csr(d_coo_desc.rows, d_coo_desc.nnz, d_coo_desc.row_index, d_row_ptr);
  info->csr_row_ptr = d_row_ptr;
}

template <typename INDEX_TYPE, typename DATA_TYPE>
void usparse::spmm::internal::coomm_conv_compute(const usparse::usparse_operation A_trans,
                                                 const usparse::usparse_operation B_trans, const DATA_TYPE alpha,
                                                 const DATA_TYPE beta,
                                                 const usparse::coo_desc<INDEX_TYPE, DATA_TYPE> d_coo_desc,
                                                 const INDEX_TYPE B_cols, const DATA_TYPE *B_dense_val,
                                                 DATA_TYPE *C_dense_val,
                                                 usparse::spmm::internal::coomm_conversion_mtx_info<INDEX_TYPE> *info,
                                                 const usparse::spmm::usparse_coomm_algorithms algo) {
  const usparse::csr_desc<INDEX_TYPE, DATA_TYPE> d_csr_desc = usparse::csr_desc<INDEX_TYPE, DATA_TYPE>(
      d_coo_desc.rows, d_coo_desc.cols, d_coo_desc.nnz, info->csr_row_ptr, d_coo_desc.col_index, d_coo_desc.values);
  if (algo == usparse::spmm::usparse_coomm_algorithms::Vector) {
    usparse::internal::adaptive_csrmm_vetor_row<INDEX_TYPE, DATA_TYPE, __WF_SIZE__>(
        A_trans, B_trans, alpha, beta, d_csr_desc, B_cols, B_dense_val, C_dense_val);
  } else {
    assert(false);
  }
}

template <typename INDEX_TYPE>
void usparse::spmm::internal::coomm_conv_clear(usparse::spmm::internal::coomm_conversion_mtx_info<INDEX_TYPE> *info) {
  // todo:
}

template void usparse::spmm::internal::coomm_conv_analysis<usparse::usparse_int, usparse::usparse_fp32>(
    const usparse::usparse_operation trans,
    const usparse::coo_desc<usparse::usparse_int, usparse::usparse_fp32> d_coo_desc,
    usparse::spmm::internal::coomm_conversion_mtx_info<usparse::usparse_int> *info,
    const usparse::spmm::usparse_coomm_algorithms algo);

template void usparse::spmm::internal::coomm_conv_analysis<usparse::usparse_int, usparse::usparse_fp64>(
    const usparse::usparse_operation trans,
    const usparse::coo_desc<usparse::usparse_int, usparse::usparse_fp64> d_coo_desc,
    usparse::spmm::internal::coomm_conversion_mtx_info<usparse::usparse_int> *info,
    const usparse::spmm::usparse_coomm_algorithms algo);

template void usparse::spmm::internal::coomm_conv_compute<usparse::usparse_int, usparse::usparse_fp32>(
    const usparse::usparse_operation A_trans, const usparse::usparse_operation B_trans,
    const usparse::usparse_fp32 alpha, const usparse::usparse_fp32 beta,
    const usparse::coo_desc<usparse::usparse_int, usparse::usparse_fp32> d_coo_desc, const usparse::usparse_int B_cols,
    const usparse::usparse_fp32 *B_dense_val, usparse::usparse_fp32 *C_dense_val,
    usparse::spmm::internal::coomm_conversion_mtx_info<usparse::usparse_int> *info,
    const usparse::spmm::usparse_coomm_algorithms algo);

template void usparse::spmm::internal::coomm_conv_compute<usparse::usparse_int, usparse::usparse_fp64>(
    const usparse::usparse_operation A_trans, const usparse::usparse_operation B_trans,
    const usparse::usparse_fp64 alpha, const usparse::usparse_fp64 beta,
    const usparse::coo_desc<usparse::usparse_int, usparse::usparse_fp64> d_coo_desc, const usparse::usparse_int B_cols,
    const usparse::usparse_fp64 *B_dense_val, usparse::usparse_fp64 *C_dense_val,
    usparse::spmm::internal::coomm_conversion_mtx_info<usparse::usparse_int> *info,
    const usparse::spmm::usparse_coomm_algorithms algo);
