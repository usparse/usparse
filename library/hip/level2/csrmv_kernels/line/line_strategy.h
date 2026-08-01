//
// Created by genshen on 2021/06/30.
//

#ifndef SPMV_ACC_SPMV_HIP_ACC_IMP_LINE_H
#define SPMV_ACC_SPMV_HIP_ACC_IMP_LINE_H

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "usparse_matrix.h"

template <typename INDEX_TYPE, typename DATA_TYPE>
void adaptive_line_sparse_spmv(int trans, const DATA_TYPE alpha, const DATA_TYPE beta,
                               const csr_desc<INDEX_TYPE, DATA_TYPE> d_csr_desc, const DATA_TYPE *x, DATA_TYPE *y);

template <typename INDEX_TYPE, typename DATA_TYPE>
void line_sparse_spmv(int trans, const DATA_TYPE alpha, const DATA_TYPE beta,
                      const csr_desc<INDEX_TYPE, DATA_TYPE> d_csr_desc, const DATA_TYPE *x, DATA_TYPE *y);

#endif // SPMV_ACC_SPMV_HIP_ACC_IMP_LINE_H
