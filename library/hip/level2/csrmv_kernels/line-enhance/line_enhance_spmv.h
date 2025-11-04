//
// Created by chu genshen on 2021/10/2.
//

#ifndef SPMV_ACC_LINE_ENHANCE_SPMV_H
#define SPMV_ACC_LINE_ENHANCE_SPMV_H

#include "usparse_matrix.h"

template <typename INDEX_TYPE, typename DATA_TYPE>
void line_enhance_sparse_spmv(int trans, const DATA_TYPE alpha, const DATA_TYPE beta,
                              const csr_desc<INDEX_TYPE, DATA_TYPE> d_csr_desc, const DATA_TYPE *x, DATA_TYPE *y);

/**
 * @tparam INDEX_TYPE type for index.
 * @tparam FPType the Type for float precision. Can be float or double.
 * set kernel template parameters adaptively by matrix characteristic.
 */
template <typename INDEX_TYPE, typename FPType>
void adaptive_enhance_sparse_spmv(int trans, const FPType alpha, const FPType beta,
                                  const csr_desc<INDEX_TYPE, FPType> d_csr_desc, const FPType *x, FPType *y);

#endif // SPMV_ACC_LINE_ENHANCE_SPMV_H
