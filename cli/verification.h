//
// Created by chu genshen on 2021/9/14.
//

#ifndef SPMV_ACC_VERIFICATION_H
#define SPMV_ACC_VERIFICATION_H

#include "building_config.h"
#include "matrix_format/sparse_format.h"

template <typename I, typename T> struct VerifyResult {
  T max_error = 0.0;
  I first_failed_at = -1;
  I failed_count = 0;
};

template <typename I, typename T> VerifyResult<I, T> verify_y(T *dy, T *hy, I N);

void verify(double *dy, double *hy, int n);

template <typename INDEX_TYPE, typename DATA_TYPE>
void host_coo_spmv(DATA_TYPE alpha, DATA_TYPE beta, const DATA_TYPE *value, const INDEX_TYPE *row_index,
                   const INDEX_TYPE *colindex, INDEX_TYPE m, INDEX_TYPE n, INDEX_TYPE nnz, const DATA_TYPE *x,
                   DATA_TYPE *y);

template <typename INDEX_TYPE, typename DATA_TYPE>
void host_coo_spmv(const DATA_TYPE *value, const INDEX_TYPE *row_index, const INDEX_TYPE *colindex, INDEX_TYPE m,
                   INDEX_TYPE n, INDEX_TYPE nnz, const DATA_TYPE *x, DATA_TYPE *y);

template <typename INDEX_TYPE, typename DATA_TYPE>
void host_csr_spmv(DATA_TYPE alpha, DATA_TYPE beta, const DATA_TYPE *value, const INDEX_TYPE *rowptr,
                   const INDEX_TYPE *colindex, INDEX_TYPE m, INDEX_TYPE n, INDEX_TYPE a, const DATA_TYPE *x,
                   DATA_TYPE *y);

template <typename INDEX_TYPE, typename DATA_TYPE>
void host_csr_spmv(const DATA_TYPE *value, const int *rowptr, const INDEX_TYPE *colindex, INDEX_TYPE m, INDEX_TYPE n,
                   INDEX_TYPE a, const DATA_TYPE *x, DATA_TYPE *y);

#ifdef gpu
void rocsparse(type_csr d_csr, dtype *dev_x, dtype *dev_y, dtype &alpha, dtype &beta);
#endif

#endif // SPMV_ACC_VERIFICATION_H
