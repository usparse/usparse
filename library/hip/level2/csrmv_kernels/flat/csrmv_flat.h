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
          int FLAT_STORE_Y_POLICY = FLAT_STORE_Y_POLICY_ATOMIC>
void adaptive_csrmv_flat(hipStream_t stream, int trans, const SCALE_TYPE alpha, const SCALE_TYPE beta,
                         const csr_mat_descr<A, NNZ, ROW, COL> d_csr_desc, const X *x, Y *y,
                         const csrmv_flat_mtx_info<BP_DATA_TYPE, Y> *info);
/**
 * call adaptive_csrmv_flat and set @tparam FLAT_STORE_Y_POLICY as FLAT_STORE_Y_POLICY_ATOMIC
 * (use atomicAdd for storing result vector y).
 */
template <typename T, typename BP_DATA_TYPE, typename A, typename X, typename Y, typename NNZ, typename ROW,
          typename COL, typename SCALE_TYPE, int BETA_Y_POLICY = BETA_Y_POLICY_AUTO>
void adaptive_csrmv_flat_atomic(hipStream_t stream, int trans, const SCALE_TYPE alpha, const SCALE_TYPE beta,
                                const csr_mat_descr<A, NNZ, ROW, COL> d_csr_desc, const X *x, Y *y,
                                const csrmv_flat_mtx_info<BP_DATA_TYPE, Y> *info);

/**
 * call adaptive_csrmv_flat and set @tparam FLAT_STORE_Y_POLICY as FLAT_STORE_Y_POLICY_SEGMENTED_SUM
 * (use segmented sum for storing result vector y).
 */
template <typename T, typename BP_DATA_TYPE, typename A, typename X, typename Y, typename NNZ, typename ROW,
          typename COL, typename SCALE_TYPE, int BETA_Y_POLICY = BETA_Y_POLICY_AUTO>
void adaptive_csrmv_flat_segmented(hipStream_t stream, int trans, const SCALE_TYPE alpha, const SCALE_TYPE beta,
                                   const csr_mat_descr<A, NNZ, ROW, COL> d_csr_desc, const X *x, Y *y,
                                   const csrmv_flat_mtx_info<BP_DATA_TYPE, Y> *info);

/**
 * the general matrix analyse api for the flat csrmv,
 * which performs matrix analyse and return analysing results in @param info.
 * The template parameter @tparam FLAT_STORE_Y_POLICY can be FLAT_STORE_Y_POLICY_ATOMIC or
 * FLAT_STORE_Y_POLICY_SEGMENTED_SUM.
 *
 * @tparam DATA_TYPE: the element data type of matrix values in csrmv.
 * @tparam NNZ_TYPE: the index type of non-zeros in csrm matrix.
 * @tparam ROW_TYPE: the index type of rows in csr matrix.
 * @tparam COLUMN_TYPE: the index type of columns in csr matrix.
 * @tparam BP_DATA_TYPE: the index type of columns in csr matrix.
 * @tparam Y_TYPE: the element data type of vector y in csrmv. The type is used to create part sum temp buffer (only
 * needed if the template parameter @tparam FLAT_STORE_Y_POLICY is FLAT_STORE_Y_POLICY_SEGMENTED_SUM).
 * @tparam FLAT_STORE_Y_POLICY: the policy of storing result vector y. It can be FLAT_STORE_Y_POLICY_ATOMIC or
 * FLAT_STORE_Y_POLICY_SEGMENTED_SUM.
 * @tparam CREAT_PART_SUM_TEMP_BUFFER: whether to create the temp buffer for part sum in the analyse stage.
 * @tparam CREAT_PART_SUM_TEMP_BUFFER: if true, create the device array for storing part sum (only create if @tparam
 * FLAT_STORE_Y_POLICY is FLAT_STORE_Y_POLICY_SEGMENTED_SUM).
 *
 * @param stream: the hip stream for running pre-processing kernels and hip memory api.
 * @param d_csr_desc: the descriptor of the csr matrix.
 * @param info: the pointer to analyse info, which stores the analyse results.
 */
template <typename DATA_TYPE, typename NNZ_TYPE, typename ROW_TYPE, typename COLUMN_TYPE, typename BP_DATA_TYPE,
          typename Y_TYPE, int FLAT_STORE_Y_POLICY, bool CREAT_PART_SUM_TEMP_BUFFER>
void flat_matrix_analyse(hipStream_t stream, const csr_mat_descr<DATA_TYPE, NNZ_TYPE, ROW_TYPE, COLUMN_TYPE> d_csr_desc,
                         csrmv_flat_mtx_info<BP_DATA_TYPE, Y_TYPE> *info);

/**
 * perform matrix analyse and return analysing results in @param info.
 *
 * @note: It is similar as the above matrix analyse api.
 * the difference is that this api set template parameter FLAT_STORE_Y_POLICY to FLAT_STORE_Y_POLICY_ATOMIC
 * (use atomicAdd for storing result vector y).
 */
template <typename DATA_TYPE, typename NNZ_TYPE, typename ROW_TYPE, typename COLUMN_TYPE, typename BP_DATA_TYPE,
          typename Y_TYPE>
void flat_matrix_analyse_atomic(hipStream_t stream,
                                const csr_mat_descr<DATA_TYPE, NNZ_TYPE, ROW_TYPE, COLUMN_TYPE> d_csr_desc,
                                csrmv_flat_mtx_info<BP_DATA_TYPE, Y_TYPE> *info);

/**
 * perform matrix analyse and return analysing results in @param info.
 *
 * @note: It is similar as the above matrix analyse api.
 * the difference is that this api set template parameter FLAT_STORE_Y_POLICY to FLAT_STORE_Y_POLICY_SEGMENTED_SUM
 * (use segmented sum for storing result vector y).
 */
template <typename DATA_TYPE, typename NNZ_TYPE, typename ROW_TYPE, typename COLUMN_TYPE, typename BP_DATA_TYPE,
          typename Y_TYPE>
void flat_matrix_analyse_segmented(hipStream_t stream,
                                   const csr_mat_descr<DATA_TYPE, NNZ_TYPE, ROW_TYPE, COLUMN_TYPE> d_csr_desc,
                                   csrmv_flat_mtx_info<BP_DATA_TYPE, Y_TYPE> *info);
/**
 * the general matrix clear api for the flat csrmv,
 * which frees the memory allocated during the analyse stage.
 */
template <typename BP_DATA_TYPE, typename Y_TYPE, int FLAT_STORE_Y_POLICY, bool CREAT_PART_SUM_TEMP_BUFFER>
void flat_matrix_clear(hipStream_t stream, csrmv_flat_mtx_info<BP_DATA_TYPE, Y_TYPE> *info);

/**
 * perform memory destruction for atomic mode.
 */
template <typename BP_DATA_TYPE, typename Y_TYPE>
void flat_matrix_clear_atomic(hipStream_t stream, csrmv_flat_mtx_info<BP_DATA_TYPE, Y_TYPE> *info);

/**
 * perform memory destruction for segmented sum mode.
 */
template <typename BP_DATA_TYPE, typename Y_TYPE>
void flat_matrix_clear_segmented(hipStream_t stream, csrmv_flat_mtx_info<BP_DATA_TYPE, Y_TYPE> *info);

/**
 * non-type version of matrix info desctory.
 */
void flat_matrix_clear(hipStream_t stream, void *info);
void flat_matrix_clear_atomic(hipStream_t stream, void *info);
void flat_matrix_clear_segmented(hipStream_t stream, void *info);

#endif // USPARSE_HIP_USPARSE_CSRMV_FLAT_H
