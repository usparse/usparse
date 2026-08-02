//
// Created by chaohu on 2021/05/25.
//

#ifndef SPMV_ACC_VECTOR_ROW_H
#define SPMV_ACC_VECTOR_ROW_H

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "usparse_matrix.h"

template <typename DATA_TYPE>
void vec_row_sparse_spmv(int trans, const DATA_TYPE alpha, const DATA_TYPE beta,
                         const csr_desc<int, DATA_TYPE> d_csr_desc, const DATA_TYPE *x, DATA_TYPE *y);

template <typename DATA_TYPE>
void adaptive_vec_row_sparse_spmv(const int nnz_block_0, const int nnz_block_1, int trans, const DATA_TYPE alpha,
                                  const DATA_TYPE beta, const csr_desc<int, DATA_TYPE> d_csr_desc, const DATA_TYPE *x,
                                  DATA_TYPE *y);

#endif // SPMV_ACC_VECTOR_ROW_H
