//
// Created by chaohu on 2021/05/25.
//

#ifndef SPMV_ACC_SPMV_HIP_ACC_IMP_LIGHT_H
#define SPMV_ACC_SPMV_HIP_ACC_IMP_LIGHT_H

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "usparse_matrix.h"

template <typename INDEX_TYPE, typename DATA_TYPE>
void light_sparse_spmv(int trans, const DATA_TYPE alpha, const DATA_TYPE beta,
                       const csr_desc<INDEX_TYPE, DATA_TYPE> d_csr_desc, const DATA_TYPE *x, DATA_TYPE *y);

#endif // SPMV_ACC_SPMV_HIP_ACC_IMP_LIGHT_H
