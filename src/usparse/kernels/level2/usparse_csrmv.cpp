//
// Created by genshen to 2025-07-09.
//

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <rocsparse/rocsparse.h>

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

// todo: the parameter passed to spmv kernel api must int. there is a conversion from uint to int.
template void usparse::device_csrmv<uint, double>(const uint nrows, const uint ncols, const uint nnz,
                                                  const uint *csr_row_ptr, const uint *csr_col_index,
                                                  const double *csr_values, const double *dev_x, double *dev_y,
                                                  rocsparse_handle *handle, rocsparse_mat_descr *descrA,
                                                  rocsparse_mat_info *info);

// todo: support float
// template void usparse::device_csrmv<uint, float>(const uint nrows, const uint ncols, const uint nnz,
//                                                  const uint *csr_row_ptr, const uint *csr_col_index,
//                                                  const float *csr_values, const float *dev_x, float *dev_y,
//                                                  rocsparse_handle *handle, rocsparse_mat_descr *descrA,
//                                                  rocsparse_mat_info *info);
