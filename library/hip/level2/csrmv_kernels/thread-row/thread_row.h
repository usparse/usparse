//
// Created by dly on 2021/4/19.
//

#ifndef SPMV_ACC_THREAD_ROW_H
#define SPMV_ACC_THREAD_ROW_H

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "usparse_matrix.h"

template <typename INDEX_TYPE, typename DATA_TYPE>
void thread_row_sparse_spmv(int trans, const DATA_TYPE alpha, const DATA_TYPE beta,
                            const csr_desc<INDEX_TYPE, DATA_TYPE> d_csr_desc, const DATA_TYPE *d_x, DATA_TYPE *d_y);

#endif // SPMV_ACC_THREAD_ROW_H
