//
// Created by chu genshen on 2021/10/2.
//

#ifndef SPMV_ACC_LINE_ENHANCE_SPMV_H
#define SPMV_ACC_LINE_ENHANCE_SPMV_H

#include "usparse_matrix.h"

template <typename INDEX_TYPE, typename DATA_TYPE>
void line_enhance_sparse_spmv(hipStream_t stream, int trans, const DATA_TYPE alpha, const DATA_TYPE beta,
                              const csr_desc<INDEX_TYPE, DATA_TYPE> d_csr_desc, const DATA_TYPE *x, DATA_TYPE *y);

/**
 * @deprecated, use the version with HIP stream instead.
 */
template <typename INDEX_TYPE, typename DATA_TYPE>
void line_enhance_sparse_spmv(int trans, const DATA_TYPE alpha, const DATA_TYPE beta,
                              const csr_desc<INDEX_TYPE, DATA_TYPE> d_csr_desc, const DATA_TYPE *x, DATA_TYPE *y);

/**
 * @tparam INDEX_TYPE type for index.
 * @tparam FPType the Type for float precision. Can be float or double.
 * set kernel template parameters adaptively by matrix characteristic.
 */
template <typename T, typename I, typename J, typename A, typename X, typename Y>
void adaptive_enhance_sparse_spmv(hipStream_t stream, int trans, const T alpha, const T beta,
                                  const csr_desc<I, A, J> d_csr_desc, const X *x, Y *y);

/**
 * @deprecated, use the version with HIP stream instead.
 * adaptive enhance spmv without HIP stream.
 */
template <typename T, typename I, typename J, typename A, typename X, typename Y>
void adaptive_enhance_sparse_spmv(int trans, const T alpha, const T beta, const csr_desc<I, A, J> d_csr_desc,
                                  const X *x, Y *y);

#endif // SPMV_ACC_LINE_ENHANCE_SPMV_H
