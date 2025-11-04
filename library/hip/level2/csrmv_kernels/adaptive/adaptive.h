//
// Created by genshen on 2021/7/28.
//

#ifndef SPMV_ACC_ADAPTIVE_H
#define SPMV_ACC_ADAPTIVE_H

#include "usparse_matrix.h"

template <typename INDEX_TYPE, typename DATA_TYPE>
void adaptive_sparse_spmv(int trans, const DATA_TYPE alpha, const DATA_TYPE beta,
                          const csr_desc<INDEX_TYPE, DATA_TYPE> h_csr_desc,
                          const csr_desc<INDEX_TYPE, DATA_TYPE> d_csr_desc, const DATA_TYPE *x, DATA_TYPE *y);

#endif // SPMV_ACC_ADAPTIVE_H
