//
// Created by chu genshen on 2021/10/2.
//

#ifndef SPMV_ACC_LINE_ENHANCE_SPMV_H
#define SPMV_ACC_LINE_ENHANCE_SPMV_H

#include "usparse_matrix.h"

/**
 * perform csrmv using line enhance algorithm with defaut parameter. (http://doi.org/10.1145/3588195.3593002).
 * @tparam COMPUTE type for csrmv computation.
 * @tparam NNZ index type for non-zeros.
 * @tparam ROW index type for matrix rows.
 * @tparam COL index type for matrix colum.
 * @tparam SCALE data type of alpha and beta in spmv: y = alpha * Ax + beta y.
 * @tparam A data type of matrix value in csr.
 * @tparam X data type of vector x in spmv: y = alpha * Ax + beta y.
 * @tparam Y data type of vector y in spmv: y = alpha * Ax + beta y.
 */
template <typename COMPUTE_TYPE, typename NNZ_TYPE, typename ROW_TYPE, typename COL_TYPE, typename SCALE, typename A,
          typename X, typename Y, int R, int ROWS_PER_BLOCK, int VEC_SIZE, int THREADS_PER_BLOCK>
void line_enhance_sparse_spmv(hipStream_t stream, int trans, const SCALE alpha, const SCALE beta,
                              const usparse::csr_mat_descr<A, NNZ_TYPE, ROW_TYPE, COL_TYPE> d_csr_desc, const X *x,
                              Y *y);

/**
 * @deprecated, use the version with HIP stream instead.
 */
template <typename INDEX_TYPE, typename DATA_TYPE>
void line_enhance_sparse_spmv(int trans, const DATA_TYPE alpha, const DATA_TYPE beta,
                              const usparse::csr_mat_descr<DATA_TYPE, INDEX_TYPE, INDEX_TYPE, INDEX_TYPE> d_csr_desc,
                              const DATA_TYPE *x, DATA_TYPE *y);

/**
 * perform csrmv using line enhance algorithm (http://doi.org/10.1145/3588195.3593002).
 * similar as api line_enhance_sparse_spmv, but it set kernel template parameters adaptively
 *   by matrix characteristic, which usully has better performance.
 * @tparam COMPUTE type for csrmv computation.
 * @tparam NNZ index type for non-zeros.
 * @tparam ROW index type for matrix rows.
 * @tparam COL index type for matrix colum.
 * @tparam SCALE data type of alpha and beta in spmv: y = alpha * Ax + beta y.
 * @tparam A data type of matrix value in csr.
 * @tparam X data type of vector x in spmv: y = alpha * Ax + beta y.
 * @tparam Y data type of vector y in spmv: y = alpha * Ax + beta y.
 */
template <typename COMPUTE, typename NNZ, typename ROW, typename COL, typename SCALE, typename A, typename X,
          typename Y>
void adaptive_enhance_sparse_spmv(hipStream_t stream, int trans, const SCALE alpha, const SCALE beta,
                                  const usparse::csr_mat_descr<A, NNZ, ROW, COL> d_csr_desc, const X *x, Y *y);

/**
 * @deprecated, use the version with HIP stream instead.
 * adaptive enhance spmv without HIP stream.
 */
template <typename COMPUTE, typename NNZ, typename ROW, typename COL, typename SCALE, typename A, typename X,
          typename Y>
void adaptive_enhance_sparse_spmv(int trans, const SCALE alpha, const SCALE beta,
                                  const usparse::csr_mat_descr<A, NNZ, ROW, COL> d_csr_desc, const X *x, Y *y);

#endif // SPMV_ACC_LINE_ENHANCE_SPMV_H
