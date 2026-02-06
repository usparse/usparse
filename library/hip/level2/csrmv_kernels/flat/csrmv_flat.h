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
#include "flat_config.h"

/**
 * the flat api with segment sum for reductionn, and without stream support and without mixed datatype.
 */
void segment_sum_flat_sparse_spmv(int trans, const double alpha, const double beta,
                                  const csr_mat_descr<double, int, int, int> h_csr_desc,
                                  const csr_mat_descr<double, int, int, int> d_csr_desc, const double *x, double *y);

template <typename T, typename BP_DATA_TYPE, typename A, typename X, typename Y, typename NNZ, typename ROW,
          typename COL, typename SCALE_TYPE, int BETA_Y_POLICY = BETA_Y_POLICY_AUTO,
          int FLAT_STORE_Y_POLICY = FLAT_STORE_Y_POLICY_UNDETERMINISTIC>
void adaptive_flat_sparse_spmv(hipStream_t stream, int trans, const SCALE_TYPE alpha, const SCALE_TYPE beta,
                               const csr_mat_descr<A, NNZ, ROW, COL> d_csr_desc, const X *x, Y *y,
                               const csrmv_flat_mtx_info<BP_DATA_TYPE, Y> *info);

/**
 * @deprecated. use the mixed datatype and hip stream version.
 *
 * the flat api without hip stream and mixed datatype support.
 */
template <typename I, typename T, int BETA_Y_POLICY = BETA_Y_POLICY_AUTO>
void adaptive_flat_sparse_spmv(int trans, const T alpha, const T beta, const csr_mat_descr<T, I, I, I> d_csr_desc,
                               const T *x, T *y, const csrmv_flat_mtx_info<I, T> *info);

/**
 * perform matrix analyse and return analysing results in @param info.
 *
 * @tparam Y_TYPE: the element data type of vector y in csrmv. It is used to create part sum temp buffer (only needed if
 * @tparam FLAT_STORE_Y_POLICY is FLAT_STORE_Y_POLICY_DETERMINISTIC).
 * @tparam CREAT_PART_SUM_TEMP_BUFFER: if true, create the device array for storing part sum (only create if @tparam
 * FLAT_STORE_Y_POLICY is FLAT_STORE_Y_POLICY_DETERMINISTIC).
 */
template <typename DATA_TYPE, typename NNZ_TYPE, typename ROW_TYPE, typename COLUMN_TYPE, typename BP_DATA_TYPE,
          typename Y_TYPE, int FLAT_STORE_Y_POLICY, bool CREAT_PART_SUM_TEMP_BUFFER>
void flat_matrix_analyse(hipStream_t stream, const csr_mat_descr<DATA_TYPE, NNZ_TYPE, ROW_TYPE, COLUMN_TYPE> d_csr_desc,
                         csrmv_flat_mtx_info<BP_DATA_TYPE, Y_TYPE> *info);

/**
 * @deprecated. use hip stream version.
 *
 * call preprocessing of flat.
 * todo: remove the default template parameters
 */
template <typename DATA_TYPE, typename NNZ_TYPE = int, typename ROW_TYPE = int, typename COLUMN_TYPE = int,
          typename BP_DATA_TYPE = ROW_TYPE>
void flat_analyse(const csr_mat_descr<DATA_TYPE, NNZ_TYPE, ROW_TYPE, COLUMN_TYPE> d_csr_desc,
                  csrmv_flat_mtx_info<BP_DATA_TYPE, DATA_TYPE> *info);

#endif // USPARSE_HIP_USPARSE_CSRMV_FLAT_H
