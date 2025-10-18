//
// Created by chu genshen on 2021/10/2.
//

#ifndef SPMV_ACC_LINE_ENHANCE_SPMV_H
#define SPMV_ACC_LINE_ENHANCE_SPMV_H

#include "usparse_matrix.h"

void line_enhance_sparse_spmv(int trans, const double alpha, const double beta, const csr_desc<int, double> d_csr_desc,
                              const double *x, double *y);

/**
 * @tparam FPType the Type for float precision. Can be float or double.
 * set kernel template parameters adaptively by matrix characteristic.
 */
template <typename FPType>
void adaptive_enhance_sparse_spmv(int trans, const FPType alpha, const FPType beta,
                                  const csr_desc<int, FPType> d_csr_desc, const FPType *x, FPType *y);

#endif // SPMV_ACC_LINE_ENHANCE_SPMV_H
