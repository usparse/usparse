//
// Created by chu genshen on 2021/10/2.
//

#ifndef SPMV_ACC_LINE_ENHANCE_SPMV_H
#define SPMV_ACC_LINE_ENHANCE_SPMV_H

#include "usparse_matrix.h"
#include "line_enhance_kernels_imp.h" // 必须有这个，否则找不到 WRAPPER 宏
#include "common/macros.h"            // 必须有这个，否则找不到 VAR_FROM_CSR_DESC
template <typename INDEX_TYPE, typename DATA_TYPE, int R, int ROWS_PER_BLOCK, int VEC_SIZE, int THREADS_PER_BLOCK>
void line_enhance_sparse_spmv(hipStream_t stream, int trans, const DATA_TYPE alpha, const DATA_TYPE beta,
                              const csr_desc<INDEX_TYPE, DATA_TYPE> d_csr_desc, const DATA_TYPE *x, DATA_TYPE *y);

/**
 * @deprecated, use the version with HIP stream instead.
 */
template <typename INDEX_TYPE, typename DATA_TYPE>
void line_enhance_sparse_spmv(int trans, const DATA_TYPE alpha, const DATA_TYPE beta,
                              const csr_desc<INDEX_TYPE, DATA_TYPE> d_csr_desc, const DATA_TYPE *x, DATA_TYPE *y);

/**
 * @tparam INDEX_TYPE type for index.
 * @tparam FPType the Type for float precision. Can be float or double.
 * set kernel template parameters adaptively by matrix characteristic.
 */
template <typename T, typename I, typename J, typename A, typename X, typename Y>
void adaptive_enhance_sparse_spmv(hipStream_t stream, int trans, const T alpha, const T beta,
                                  const csr_desc<I, A, J> d_csr_desc, const X *x, Y *y);

/**
 * @deprecated, use the version with HIP stream instead.
 * adaptive enhance spmv without HIP stream.
 */
template <typename T, typename I, typename J, typename A, typename X, typename Y>
void adaptive_enhance_sparse_spmv(int trans, const T alpha, const T beta, const csr_desc<I, A, J> d_csr_desc,
                                  const X *x, Y *y);
template <typename INDEX_TYPE, typename DATA_TYPE, int R, int ROWS_PER_BLOCK, int VEC_SIZE, int THREADS_PER_BLOCK>
void line_enhance_sparse_spmv(hipStream_t stream, int trans, const DATA_TYPE alpha, const DATA_TYPE beta,
                              const csr_desc<INDEX_TYPE, DATA_TYPE> d_csr_desc, const DATA_TYPE *x, DATA_TYPE *y) {
  VAR_FROM_CSR_DESC(DATA_TYPE, d_csr_desc)

  constexpr int REDUCE_OPTION = VEC_SIZE == 0 ? LE_REDUCE_OPTION_DIRECT : LE_REDUCE_OPTION_VEC;

  int BLOCKS = m / ROWS_PER_BLOCK + (m % ROWS_PER_BLOCK == 0 ? 0 : 1);
  LINE_ENHANCE_KERNEL_WRAPPER(stream, DATA_TYPE, INDEX_TYPE, INDEX_TYPE, DATA_TYPE, DATA_TYPE, DATA_TYPE, REDUCE_OPTION,
                              ROWS_PER_BLOCK, VEC_SIZE, R, BLOCKS, THREADS_PER_BLOCK);
}

#endif // SPMV_ACC_LINE_ENHANCE_SPMV_H
