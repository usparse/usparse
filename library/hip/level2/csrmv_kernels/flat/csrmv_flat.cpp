//
// Created by genshen on 2021/7/15.
//

#include <cstdint>
#include <iostream>

#include "../../../level1/usparse_blas_level1.h"
#include "common/macros.h"
#include "building_config.h"
#include "flat_config.h"
#include "flat_kernels_imp_multi_pass.hpp"
#include "flat_kernels_imp_one_pass.hpp"
#include "flat_kernels_imp_pre_calc.hpp"
#include "csrmv_flat.h"

template <typename HIP_BLOCK_NUM_TYPE, typename SHARE_MEM_INDEX_TYPE, typename BP_INDEX_TYPE, typename BP_DATA_TYPE,
          typename A_TYPE, typename X_TYPE, typename Y_TYPE, typename COMPUTE_MULTI_TYPE, typename COMPUTE_SUM_TYPE,
          typename SCALE_DATA_TYPE, typename NNZ_TYPE, typename ROW_TYPE, typename COLUMN_TYPE, int R,
          int REDUCE_OPTION, int REDUCE_VEC_SIZE, int BLOCKS, int THREADS_PER_BLOCK>
inline void flat_multi_pass_sparse_spmv(hipStream_t stream, int trans, const SCALE_DATA_TYPE alpha,
                                        const SCALE_DATA_TYPE beta, ROW_TYPE m, COLUMN_TYPE n, NNZ_TYPE nnz,
                                        const NNZ_TYPE *rowptr, const COLUMN_TYPE *colindex, const A_TYPE *value,
                                        const X_TYPE *x, Y_TYPE *y) {
  BP_DATA_TYPE *break_points;
  // the nnz is rowptr[m], in one round, it can process about `blocks * R * threads_per_block` nnz.
  const NNZ_TYPE total_rounds =
      nnz / (R * THREADS_PER_BLOCK * BLOCKS) + (nnz % (R * THREADS_PER_BLOCK * BLOCKS) == 0 ? 0 : 1);
  // each round and each block both have a break point.
  const BP_INDEX_TYPE break_points_len = total_rounds * BLOCKS + 1;
  hipMallocAsync((void **)&break_points, break_points_len * sizeof(BP_DATA_TYPE), stream);
  hipMemsetAsync(break_points, 0, break_points_len * sizeof(BP_DATA_TYPE), stream);

  (pre_calc_break_point<ROW_TYPE, NNZ_TYPE, BP_DATA_TYPE, R * THREADS_PER_BLOCK>)<<<1024, 512, 0, stream>>>(
      rowptr, m, break_points, break_points_len);

  (csrmv_flat_multi_pass_kernel<__WF_SIZE__, R, REDUCE_OPTION, REDUCE_VEC_SIZE, BLOCKS, THREADS_PER_BLOCK,
                                SHARE_MEM_INDEX_TYPE, BP_INDEX_TYPE, BP_DATA_TYPE, SCALE_DATA_TYPE, A_TYPE,
                                COMPUTE_MULTI_TYPE, COMPUTE_SUM_TYPE, X_TYPE, Y_TYPE, NNZ_TYPE, ROW_TYPE,
                                COLUMN_TYPE>)<<<(BLOCKS), THREADS_PER_BLOCK, 0, stream>>>(
      m, alpha, beta, rowptr, break_points, colindex, value, x, y);
  hipFreeAsync(break_points, stream);
}

template <typename DATA_TYPE, typename NNZ_TYPE, typename ROW_TYPE, typename COLUMN_TYPE, typename BP_DATA_TYPE>
void flat_matrix_analyse(hipStream_t stream, const csr_mat_descr<DATA_TYPE, NNZ_TYPE, ROW_TYPE, COLUMN_TYPE> d_csr_desc,
                         BP_DATA_TYPE **break_points) {
  EXPAND_FROM_CSR_MAT_DESCR(DATA_TYPE, NNZ_TYPE, ROW_TYPE, COLUMN_TYPE, d_csr_desc)

  constexpr int R = 2;
  constexpr int THREADS_PER_BLOCK = 512;
  const int HIP_BLOCKS = nnz / (R * THREADS_PER_BLOCK) + ((nnz % (R * THREADS_PER_BLOCK) == 0) ? 0 : 1);
  constexpr int TOTAL_ROUNDS = 1;
  const NNZ_TYPE break_points_len = static_cast<NNZ_TYPE>(TOTAL_ROUNDS * HIP_BLOCKS + 1);

  hipMallocAsync((void **)break_points, break_points_len * sizeof(BP_DATA_TYPE), stream);
  hipMemsetAsync(*break_points, 0, break_points_len * sizeof(BP_DATA_TYPE), stream);
  (pre_calc_break_point<ROW_TYPE, NNZ_TYPE, BP_DATA_TYPE, R * THREADS_PER_BLOCK>)<<<1024, 512, 0, stream>>>(
      rowptr, m, *break_points, break_points_len);
}

// just for backward compatibility.
template <typename DATA_TYPE, typename NNZ_TYPE, typename ROW_TYPE, typename COLUMN_TYPE, typename BP_DATA_TYPE>
void flat_analyse(const csr_mat_descr<DATA_TYPE, NNZ_TYPE, ROW_TYPE, COLUMN_TYPE> d_csr_desc,
                  csrmv_flat_mtx_info<BP_DATA_TYPE> *info) {
  hipStream_t stream = hipStreamDefault;
  flat_matrix_analyse<DATA_TYPE, NNZ_TYPE, ROW_TYPE, COLUMN_TYPE, BP_DATA_TYPE>(stream, d_csr_desc,
                                                                                &(info->break_points));
}

template <typename HIP_BLOCK_NUM_TYPE, typename SHARE_MEM_INDEX_TYPE, typename BP_INDEX_TYPE, typename BP_DATA_TYPE,
          typename A_TYPE, typename X_TYPE, typename Y_TYPE, typename COMPUTE_MULTI_TYPE, typename COMPUTE_SUM_TYPE,
          typename SCALE_DATA_TYPE, typename NNZ_TYPE, typename ROW_TYPE, typename COLUMN_TYPE, int R,
          int REDUCE_OPTION, int REDUCE_VEC_SIZE, int THREADS_PER_BLOCK>
inline void flat_one_pass_sparse_spmv(hipStream_t stream, int trans, const SCALE_DATA_TYPE alpha,
                                      const SCALE_DATA_TYPE beta, ROW_TYPE m, COLUMN_TYPE n, NNZ_TYPE nnz,
                                      const NNZ_TYPE *rowptr, const COLUMN_TYPE *colindex, const A_TYPE *value,
                                      const X_TYPE *x, Y_TYPE *y, const BP_DATA_TYPE *break_points) {
  // each block can process `R*THREADS_PER_BLOCK` non-zeros.
  const HIP_BLOCK_NUM_TYPE HIP_BLOCKS = nnz / (R * THREADS_PER_BLOCK) + ((nnz % (R * THREADS_PER_BLOCK) == 0) ? 0 : 1);

  (csrmv_flat_one_pass_kernel<__WF_SIZE__, R, REDUCE_OPTION, REDUCE_VEC_SIZE, THREADS_PER_BLOCK, SHARE_MEM_INDEX_TYPE,
                              BP_INDEX_TYPE, BP_DATA_TYPE, SCALE_DATA_TYPE, A_TYPE, COMPUTE_MULTI_TYPE,
                              COMPUTE_SUM_TYPE, X_TYPE, Y_TYPE, NNZ_TYPE, ROW_TYPE,
                              COLUMN_TYPE>)<<<(HIP_BLOCKS), (THREADS_PER_BLOCK), 0, stream>>>(
      m, alpha, beta, rowptr, break_points, colindex, value, x, y);
  return;
}

// template for backward compatibility.
template <typename INDEX_TYPE, typename DATA_TYPE>
void flat_sparse_spmv(int trans, const DATA_TYPE alpha, const DATA_TYPE beta,
                      const csr_mat_descr<DATA_TYPE, INDEX_TYPE, INDEX_TYPE, INDEX_TYPE> h_csr_desc,
                      const csr_mat_descr<DATA_TYPE, INDEX_TYPE, INDEX_TYPE, INDEX_TYPE> d_csr_desc, const DATA_TYPE *x,
                      DATA_TYPE *y, const csrmv_flat_mtx_info<INDEX_TYPE> *info) {
  hipStream_t stream = hipStreamDefault;
  adaptive_flat_sparse_spmv<INDEX_TYPE, DATA_TYPE>(trans, alpha, beta, d_csr_desc, x, y, info, stream);
}

template <typename INDEX_TYPE, typename DATA_TYPE>
void segment_sum_flat_sparse_spmv(int trans, const DATA_TYPE alpha, const DATA_TYPE beta,
                                  const csr_mat_descr<DATA_TYPE, INDEX_TYPE, INDEX_TYPE, INDEX_TYPE> h_csr_desc,
                                  const csr_mat_descr<DATA_TYPE, INDEX_TYPE, INDEX_TYPE, INDEX_TYPE> d_csr_desc,
                                  const DATA_TYPE *x, DATA_TYPE *y) {
  EXPAND_FROM_CSR_MAT_DESCR(DATA_TYPE, INDEX_TYPE, INDEX_TYPE, INDEX_TYPE, d_csr_desc)

  constexpr int R = 2;
  constexpr int THREADS_PER_BLOCK = 512;
  constexpr int blocks = 512;
  constexpr int RED_OPT = FLAT_REDUCE_OPTION_SEGMENT_SUM;

  hipStream_t stream = hipStreamDefault;
  flat_multi_pass_sparse_spmv<INDEX_TYPE, INDEX_TYPE, INDEX_TYPE, INDEX_TYPE, DATA_TYPE, DATA_TYPE, DATA_TYPE,
                              DATA_TYPE, DATA_TYPE, DATA_TYPE, INDEX_TYPE, INDEX_TYPE, INDEX_TYPE, R, RED_OPT, 2,
                              blocks, THREADS_PER_BLOCK>(stream, trans, alpha, beta, m, n, nnz, rowptr, colindex, value,
                                                         x, y);
}

template <typename Y_TYPE, typename SCALE_DATA_TYPE, typename INDEX_TYPE, int BETA_Y_POLICY>
inline void flat_set_beta_y(hipStream_t stream, Y_TYPE *y, const INDEX_TYPE m, const SCALE_DATA_TYPE beta) {
  if (BETA_Y_POLICY == BETA_Y_POLICY_AUTO) {
    // set beta*y.
    if (beta == static_cast<SCALE_DATA_TYPE>(0.0)) {
      // set vector y to 0. because: y = A*x + 0.0 *y.
      usparse::device_scale_set<INDEX_TYPE, Y_TYPE>(static_cast<Y_TYPE>(0.0), y, m, stream);
    } else if (beta == static_cast<SCALE_DATA_TYPE>(1.0)) {
      // do nothing. because flat is based on atomicAdd.
    } else {
      usparse::device_dscal<INDEX_TYPE, Y_TYPE>(beta, y, m, stream);
    }
  } else if (BETA_Y_POLICY == BETA_Y_POLICY_ZERO) {
    usparse::device_scale_set<INDEX_TYPE, Y_TYPE>(static_cast<Y_TYPE>(0.0), y, m, stream);
  } else if (BETA_Y_POLICY == BETA_Y_POLICY_ONE) {
    return; // nothing to do.
  } else {  // BETA_Y_POLICY == BETA_Y_POLICY_OTHER
    usparse::device_dscal<INDEX_TYPE, Y_TYPE>(beta, y, m, stream);
  }
}

/**
 * set flat kernel template parameters adaptively.
 * currently, it only support adaptive template parameters on one-pass flat method.
 * \note: if config item FLAT_ONE_PASS_ADAPTIVE is set to false, one-pass adaptive will be disabled.
 */
template <typename T, typename BP_DATA_TYPE, typename A, typename X, typename Y, typename NNZ, typename ROW,
          typename COL, typename SCALE_TYPE, int BETA_Y_POLICY>
void adaptive_flat_sparse_spmv(hipStream_t stream, int trans, const SCALE_TYPE alpha, const SCALE_TYPE beta,
                               const csr_mat_descr<A, NNZ, ROW, COL> d_csr_desc, const X *x, Y *y,
                               const csrmv_flat_mtx_info<BP_DATA_TYPE> *info) {
  EXPAND_FROM_CSR_MAT_DESCR(A, NNZ, ROW, COL, d_csr_desc)

  flat_set_beta_y<Y, SCALE_TYPE, ROW, BETA_Y_POLICY>(stream, y, m, beta);

  constexpr int R = 2;
  constexpr int THREADS_PER_BLOCK = 512;

  using COMPUTE_MULTI_TYPE = T;
  using COMPUTE_SUM_TYPE = T;

  using HIP_BLOCK_NUM_TYPE = int;
  using SHARE_MEM_INDEX_TYPE = int;
  using BP_INDEX_TYPE = int;

  // for non one-pass.
  if (!FLAT_ONE_PASS) {
    constexpr HIP_BLOCK_NUM_TYPE blocks = 512;
    constexpr int RED_OPT = FLAT_REDUCE_OPTION_VEC;

    flat_multi_pass_sparse_spmv<HIP_BLOCK_NUM_TYPE, SHARE_MEM_INDEX_TYPE, BP_INDEX_TYPE, BP_DATA_TYPE, A, X, Y,
                                COMPUTE_MULTI_TYPE, COMPUTE_SUM_TYPE, SCALE_TYPE, NNZ, ROW, COL, R, RED_OPT, 2, blocks,
                                THREADS_PER_BLOCK>(stream, trans, alpha, beta, m, n, nnz, rowptr, colindex, value, x,
                                                   y);
    return;
  }

  BP_DATA_TYPE *break_points = info->break_points;
  if (!FLAT_ONE_PASS_ADAPTIVE) {
    constexpr int RED_OPT = FLAT_REDUCE_OPTION_VEC;
    constexpr int VEC_SIZE = 4;
    flat_one_pass_sparse_spmv<HIP_BLOCK_NUM_TYPE, SHARE_MEM_INDEX_TYPE, BP_INDEX_TYPE, BP_DATA_TYPE, A, X, Y,
                              COMPUTE_MULTI_TYPE, COMPUTE_SUM_TYPE, SCALE_TYPE, NNZ, ROW, COL, R, RED_OPT, VEC_SIZE,
                              THREADS_PER_BLOCK>(stream, trans, alpha, beta, m, n, nnz, rowptr, colindex, value, x, y,
                                                 break_points);
    return;
  }

  // divide the matrix into 2 parts and get the max nnz per row of each part.
  NNZ avg_block_nnz_max = nnz / m;
  if (avg_block_nnz_max <= 32) {
    // use single thread to reduce if average nnz of row is less than 32.
    constexpr int REDUCE_OPT = FLAT_REDUCE_OPTION_DIRECT;
    constexpr int VEC_SIZE = 1;
    flat_one_pass_sparse_spmv<HIP_BLOCK_NUM_TYPE, SHARE_MEM_INDEX_TYPE, BP_INDEX_TYPE, BP_DATA_TYPE, A, X, Y,
                              COMPUTE_MULTI_TYPE, COMPUTE_SUM_TYPE, SCALE_TYPE, NNZ, ROW, COL, R, REDUCE_OPT, VEC_SIZE,
                              THREADS_PER_BLOCK>(stream, trans, alpha, beta, m, n, nnz, rowptr, colindex, value, x, y,
                                                 break_points);
  } else if (avg_block_nnz_max <= 64) {
    constexpr int REDUCE_OPT = FLAT_REDUCE_OPTION_VEC;
    // use more threads (4 threads) in vector to reduce if average nnz of row is larger than 32.
    constexpr int VEC_SIZE = 4;
    flat_one_pass_sparse_spmv<HIP_BLOCK_NUM_TYPE, SHARE_MEM_INDEX_TYPE, BP_INDEX_TYPE, BP_DATA_TYPE, A, X, Y,
                              COMPUTE_MULTI_TYPE, COMPUTE_SUM_TYPE, SCALE_TYPE, NNZ, ROW, COL, R, REDUCE_OPT, VEC_SIZE,
                              THREADS_PER_BLOCK>(stream, trans, alpha, beta, m, n, nnz, rowptr, colindex, value, x, y,
                                                 break_points);
  } else {
    constexpr int REDUCE_OPT = FLAT_REDUCE_OPTION_VEC;
    // use more threads (16 threads) in vector to reduce if average nnz of row is larger than 64.
    constexpr int VEC_SIZE = 16;
    flat_one_pass_sparse_spmv<HIP_BLOCK_NUM_TYPE, SHARE_MEM_INDEX_TYPE, BP_INDEX_TYPE, BP_DATA_TYPE, A, X, Y,
                              COMPUTE_MULTI_TYPE, COMPUTE_SUM_TYPE, SCALE_TYPE, NNZ, ROW, COL, R, REDUCE_OPT, VEC_SIZE,
                              THREADS_PER_BLOCK>(stream, trans, alpha, beta, m, n, nnz, rowptr, colindex, value, x, y,
                                                 break_points);
  }
}

template <typename I, typename T, int BETA_Y_POLICY>
void adaptive_flat_sparse_spmv(int trans, const T alpha, const T beta, const csr_mat_descr<T, I, I, I> d_csr_desc,
                               const T *x, T *y, const csrmv_flat_mtx_info<I> *info) {
  hipStream_t stream = hipStreamDefault;
  adaptive_flat_sparse_spmv<T, I, T, T, T, I, I, I, T, BETA_Y_POLICY>(stream, trans, alpha, beta, d_csr_desc, x, y,
                                                                      info);
}

#define FLAT_SPMV_INSTANTIATE_MIXED(T, BP, A, X, Y, NNZ, ROW, COL, SCALE)                                              \
  template void adaptive_flat_sparse_spmv<T, BP, A, X, Y, NNZ, ROW, COL, SCALE, BETA_Y_POLICY_AUTO>(                   \
      hipStream_t stream, int trans, const SCALE alpha, const SCALE beta,                                              \
      const csr_mat_descr<A, NNZ, ROW, COL> d_csr_desc, const X *x, Y *y, const csrmv_flat_mtx_info<BP> *info);        \
  template void adaptive_flat_sparse_spmv<T, BP, A, X, Y, NNZ, ROW, COL, SCALE, BETA_Y_POLICY_ZERO>(                   \
      hipStream_t stream, int trans, const SCALE alpha, const SCALE beta,                                              \
      const csr_mat_descr<A, NNZ, ROW, COL> d_csr_desc, const X *x, Y *y, const csrmv_flat_mtx_info<BP> *info);        \
  template void adaptive_flat_sparse_spmv<T, BP, A, X, Y, NNZ, ROW, COL, SCALE, BETA_Y_POLICY_ONE>(                    \
      hipStream_t stream, int trans, const SCALE alpha, const SCALE beta,                                              \
      const csr_mat_descr<A, NNZ, ROW, COL> d_csr_desc, const X *x, Y *y, const csrmv_flat_mtx_info<BP> *info);        \
  template void adaptive_flat_sparse_spmv<T, BP, A, X, Y, NNZ, ROW, COL, SCALE, BETA_Y_POLICY_OTHER>(                  \
      hipStream_t stream, int trans, const SCALE alpha, const SCALE beta,                                              \
      const csr_mat_descr<A, NNZ, ROW, COL> d_csr_desc, const X *x, Y *y, const csrmv_flat_mtx_info<BP> *info);

FLAT_SPMV_INSTANTIATE_MIXED(usparse::usparse_fp64, usparse::usparse_int, usparse::usparse_fp64, usparse::usparse_fp64,
                            usparse::usparse_fp64, usparse::usparse_int, usparse::usparse_int, usparse::usparse_int,
                            usparse::usparse_fp64)
FLAT_SPMV_INSTANTIATE_MIXED(usparse::usparse_fp32, usparse::usparse_int, usparse::usparse_fp32, usparse::usparse_fp32,
                            usparse::usparse_fp32, usparse::usparse_int, usparse::usparse_int, usparse::usparse_int,
                            usparse::usparse_fp32)
FLAT_SPMV_INSTANTIATE_MIXED(usparse::usparse_fpc32, usparse::usparse_int, usparse::usparse_fpc32,
                            usparse::usparse_fpc32, usparse::usparse_fpc32, usparse::usparse_int, usparse::usparse_int,
                            usparse::usparse_int, usparse::usparse_fpc32)
FLAT_SPMV_INSTANTIATE_MIXED(usparse::usparse_fpc64, usparse::usparse_int, usparse::usparse_fpc64,
                            usparse::usparse_fpc64, usparse::usparse_fpc64, usparse::usparse_int, usparse::usparse_int,
                            usparse::usparse_int, usparse::usparse_fpc64)

#define FLAT_SPMV_INSTANTIATE_DEPRECATED(I, T)                                                                         \
  template void adaptive_flat_sparse_spmv<I, T, BETA_Y_POLICY_AUTO>(                                                   \
      int trans, const T alpha, const T beta, const csr_mat_descr<T, I, I, I> d_csr_desc, const T *x, T *y,            \
      const csrmv_flat_mtx_info<I> *info);                                                                             \
  template void adaptive_flat_sparse_spmv<I, T, BETA_Y_POLICY_ZERO>(                                                   \
      int trans, const T alpha, const T beta, const csr_mat_descr<T, I, I, I> d_csr_desc, const T *x, T *y,            \
      const csrmv_flat_mtx_info<I> *info);                                                                             \
  template void adaptive_flat_sparse_spmv<I, T, BETA_Y_POLICY_ONE>(                                                    \
      int trans, const T alpha, const T beta, const csr_mat_descr<T, I, I, I> d_csr_desc, const T *x, T *y,            \
      const csrmv_flat_mtx_info<I> *info);                                                                             \
  template void adaptive_flat_sparse_spmv<I, T, BETA_Y_POLICY_OTHER>(                                                  \
      int trans, const T alpha, const T beta, const csr_mat_descr<T, I, I, I> d_csr_desc, const T *x, T *y,            \
      const csrmv_flat_mtx_info<I> *info);

FLAT_SPMV_INSTANTIATE_DEPRECATED(usparse::usparse_int, usparse::usparse_fp64)
FLAT_SPMV_INSTANTIATE_DEPRECATED(usparse::usparse_int, usparse::usparse_fp32)
FLAT_SPMV_INSTANTIATE_DEPRECATED(usparse::usparse_int, usparse::usparse_fpc32)
FLAT_SPMV_INSTANTIATE_DEPRECATED(usparse::usparse_int, usparse::usparse_fpc64)

template <typename DATA_TYPE, typename NNZ_TYPE, typename ROW_TYPE, typename COLUMN_TYPE, typename BP_DATA_TYPE>
void flat_matrix_analyse(hipStream_t stream, const csr_mat_descr<DATA_TYPE, NNZ_TYPE, ROW_TYPE, COLUMN_TYPE> d_csr_desc,
                         BP_DATA_TYPE **break_points);

#define FLAT_SPMV_ANALYSE_INSTANTIATE(A, NNZ, ROW, COL, BP)                                                            \
  template void flat_analyse<A, NNZ, ROW, COL, BP>(const csr_mat_descr<A, NNZ, ROW, COL> d_csr_desc,                   \
                                                   csrmv_flat_mtx_info<BP> *info);

FLAT_SPMV_ANALYSE_INSTANTIATE(usparse::usparse_fp64, usparse::usparse_int, usparse::usparse_int, usparse::usparse_int,
                              usparse::usparse_int);
FLAT_SPMV_ANALYSE_INSTANTIATE(usparse::usparse_fp32, usparse::usparse_int, usparse::usparse_int, usparse::usparse_int,
                              usparse::usparse_int);
FLAT_SPMV_ANALYSE_INSTANTIATE(usparse::usparse_fpc64, usparse::usparse_int, usparse::usparse_int, usparse::usparse_int,
                              usparse::usparse_int);
FLAT_SPMV_ANALYSE_INSTANTIATE(usparse::usparse_fpc32, usparse::usparse_int, usparse::usparse_int, usparse::usparse_int,
                              usparse::usparse_int);
FLAT_SPMV_ANALYSE_INSTANTIATE(int8_t, usparse::usparse_int, usparse::usparse_int, usparse::usparse_int,
                              usparse::usparse_int);

#undef FLAT_SPMV_INSTANTIATE_MIXED
#undef FLAT_SPMV_INSTANTIATE_DEPRECATED
#undef FLAT_SPMV_ANALYSE_INSTANTIATE
