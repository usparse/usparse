//
// Created by genshen on 2021/07/06.
//

#ifndef USPARSE_HIP_USPARSE_CSRMV_FLAT_H
#define USPARSE_HIP_USPARSE_CSRMV_FLAT_H

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "usparse_matrix.h"
#include "flat_mtx_info.h"
#include "../../spmv_api_config.hpp"

/**
 * the flat api with segment sum for reductionn, and without stream support and without mixed datatype.
 */
void segment_sum_flat_sparse_spmv(int trans, const double alpha, const double beta,
                                  const csr_mat_descr<double, int, int, int> h_csr_desc,
                                  const csr_mat_descr<double, int, int, int> d_csr_desc, const double *x, double *y);

template <typename T, typename BP_DATA_TYPE, typename A, typename X, typename Y, typename NNZ, typename ROW,
          typename COL, typename SCALE_TYPE, int BETA_Y_POLICY = BETA_Y_POLICY_AUTO>
void adaptive_flat_sparse_spmv(hipStream_t stream, int trans, const SCALE_TYPE alpha, const SCALE_TYPE beta,
                               const csr_mat_descr<A, NNZ, ROW, COL> d_csr_desc, const X *x, Y *y,
                               const csrmv_flat_mtx_info *info);

/**
 * @deprecated. use the mixed datatype and hip stream version.
 *
 * the flat api without hip stream and mixed datatype support.
 */
template <typename I, typename T, int BETA_Y_POLICY = BETA_Y_POLICY_AUTO>
void adaptive_flat_sparse_spmv(int trans, const T alpha, const T beta, const csr_mat_descr<T, I, I, I> d_csr_desc,
                               const T *x, T *y, const csrmv_flat_mtx_info *info);

/**
 * perform matrix analyse and return analysing results in @param info.
 *
 */
template <typename DATA_TYPE, typename NNZ_TYPE, typename ROW_TYPE, typename COLUMN_TYPE, typename BP_DATA_TYPE>
void flat_matrix_analyse(hipStream_t stream, const csr_mat_descr<DATA_TYPE, NNZ_TYPE, ROW_TYPE, COLUMN_TYPE> d_csr_desc,
                         csrmv_flat_mtx_info *info);

/**
 * @deprecated. use hip stream version.
 *
 * call preprocessing of flat.
 * todo: remove the default template parameters
 */
template <typename DATA_TYPE, typename NNZ_TYPE = int, typename ROW_TYPE = int, typename COLUMN_TYPE = int,
          typename BP_DATA_TYPE = ROW_TYPE>
void flat_analyse(const csr_mat_descr<DATA_TYPE, NNZ_TYPE, ROW_TYPE, COLUMN_TYPE> d_csr_desc,
                  csrmv_flat_mtx_info *info);

#endif // USPARSE_HIP_USPARSE_CSRMV_FLAT_H
