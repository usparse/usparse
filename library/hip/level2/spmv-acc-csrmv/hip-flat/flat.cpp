//
// Created by genshen on 2021/7/15.
//

#include <iostream>

#include "../../../level1/usparse_blas_level1.h"
#include "common/macros.h"
#include "building_config.h"
#include "flat_config.h"
#include "spmv_hip_acc_imp.h"

template <typename INDEX_TYPE, typename DATA_TYPE, int R, int REDUCE_OPTION, int REDUCE_VEC_SIZE, int BLOCKS,
          int THREADS_PER_BLOCK>
inline void flat_multi_pass_sparse_spmv(int trans, const DATA_TYPE alpha, const DATA_TYPE beta, int m, int n, int nnz,
                                        const int *rowptr, const int *colindex, const DATA_TYPE *value,
                                        const DATA_TYPE *x, DATA_TYPE *y) {
  int *break_points;
  // the nnz is rowptr[m], in one round, it can process about `blocks * R * threads_per_block` nnz.
  const int total_rounds =
      nnz / (R * THREADS_PER_BLOCK * BLOCKS) + (nnz % (R * THREADS_PER_BLOCK * BLOCKS) == 0 ? 0 : 1);
  // each round and each block both have a break point.
  const int break_points_len = total_rounds * BLOCKS + 1;
  hipMalloc((void **)&break_points, break_points_len * sizeof(int));
  hipMemset(break_points, 0, break_points_len * sizeof(int));

  (pre_calc_break_point<R * THREADS_PER_BLOCK, BLOCKS, int>)<<<1024, 512>>>(rowptr, m, break_points, break_points_len);
  FLAT_KERNEL_WRAPPER(INDEX_TYPE, DATA_TYPE, R, REDUCE_OPTION, REDUCE_VEC_SIZE, BLOCKS, THREADS_PER_BLOCK);
}

template <typename DATA_TYPE> void flat_matrix_analyse(const csr_desc<int, DATA_TYPE> d_csr_desc, int **break_points) {
  VAR_FROM_CSR_DESC(DATA_TYPE, d_csr_desc)
  const int nnz = d_csr_desc.nnz;
  const int n = d_csr_desc.cols;

  constexpr int R = 2;
  constexpr int THREADS_PER_BLOCK = 512;
  const int HIP_BLOCKS = nnz / (R * THREADS_PER_BLOCK) + ((nnz % (R * THREADS_PER_BLOCK) == 0) ? 0 : 1);
  constexpr int TOTAL_ROUNDS = 1;
  const int break_points_len = TOTAL_ROUNDS * HIP_BLOCKS + 1;

  hipMalloc((void **)break_points, break_points_len * sizeof(int));
  hipMemset(*break_points, 0, break_points_len * sizeof(int));
  (pre_calc_break_point<R * THREADS_PER_BLOCK, 0, int>)<<<1024, 512>>>(rowptr, m, *break_points, break_points_len);
}

// just for backward compatibility.
template <typename DATA_TYPE> void flat_analyse(const csr_desc<int, DATA_TYPE> d_csr_desc, csrmv_flat_mtx_info *info) {
  flat_matrix_analyse(d_csr_desc, &(info->break_points));
}

template <typename INDEX_TYPE, typename DATA_TYPE, int R, int REDUCE_OPTION, int REDUCE_VEC_SIZE, int THREADS_PER_BLOCK>
inline void flat_one_pass_sparse_spmv(int trans, const DATA_TYPE alpha, const DATA_TYPE beta, int m, int n, int nnz,
                                      const int *rowptr, const int *colindex, const DATA_TYPE *value,
                                      const DATA_TYPE *x, DATA_TYPE *y, const int *break_points) {
  // each block can process `R*THREADS_PER_BLOCK` non-zeros.
  const int HIP_BLOCKS = nnz / (R * THREADS_PER_BLOCK) + ((nnz % (R * THREADS_PER_BLOCK) == 0) ? 0 : 1);
  FLAT_KERNEL_ONE_PASS_WRAPPER(INDEX_TYPE, DATA_TYPE, R, REDUCE_OPTION, REDUCE_VEC_SIZE, HIP_BLOCKS, THREADS_PER_BLOCK);
}

template <typename INDEX_TYPE, typename DATA_TYPE>
void flat_sparse_spmv(int trans, const DATA_TYPE alpha, const DATA_TYPE beta, const csr_desc<int, DATA_TYPE> h_csr_desc,
                      const csr_desc<INDEX_TYPE, DATA_TYPE> d_csr_desc, const DATA_TYPE *x, DATA_TYPE *y,
                      const csrmv_flat_mtx_info *info) {
  adaptive_flat_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y, info);
}

template <typename INDEX_TYPE, typename DATA_TYPE>
void segment_sum_flat_sparse_spmv(int trans, const DATA_TYPE alpha, const DATA_TYPE beta,
                                  const csr_desc<INDEX_TYPE, DATA_TYPE> h_csr_desc,
                                  const csr_desc<int, DATA_TYPE> d_csr_desc, const double *x, double *y) {
  VAR_FROM_CSR_DESC(DATA_TYPE, d_csr_desc)
  const int nnz = d_csr_desc.nnz;
  const int n = d_csr_desc.cols;

  constexpr int R = 2;
  constexpr int THREADS_PER_BLOCK = 512;
  constexpr int blocks = 512;
  constexpr int RED_OPT = FLAT_REDUCE_OPTION_SEGMENT_SUM;
  flat_multi_pass_sparse_spmv<INDEX_TYPE, DATA_TYPE, R, RED_OPT, 2, blocks, THREADS_PER_BLOCK>(trans, alpha, beta, m, n, nnz,
                                                                                   rowptr, colindex, value, x, y);
}

template <typename DATA_TYPE, typename INDEX_TYPE, int BEAT_Y_POLICY>
inline void flat_set_beta_y(DATA_TYPE *y, const INDEX_TYPE m, const DATA_TYPE beta) {
  if (BEAT_Y_POLICY == BEAT_Y_POLICY_AUTO) {
    // set beta*y.
    if (beta == 0.0) {
      // set vector y to 0. because: y = A*x + 0.0 *y.
      usparse::device_scale_set<INDEX_TYPE, DATA_TYPE>(0.0, y, m);
    } else if (beta == 1.0) {
      // do nothing. because flat is based on atomicAdd.
    } else {
      usparse::device_dscal(beta, y, m);
    }
  } else if (BEAT_Y_POLICY == BEAT_Y_POLICY_ZERO) {
    usparse::device_scale_set<INDEX_TYPE, DATA_TYPE>(0.0, y, m);
  } else if (BEAT_Y_POLICY == BEAT_Y_POLICY_ONE) {
    return; // nothing to do.
  } else {  // BEAT_Y_POLICY == BEAT_Y_POLICY_OTHER
    usparse::device_dscal(beta, y, m);
  }
}

/**
 * set flat kernel template parameters adaptively.
 * currently, it only support adaptive template parameters on one-pass flat method.
 * \note: if config item FLAT_ONE_PASS_ADAPTIVE is set to false, one-pass adaptive will be disabled.
 */
template <typename T, int BEAT_Y_POLICY>
void adaptive_flat_sparse_spmv(int trans, const T alpha, const T beta, const csr_desc<int, T> d_csr_desc, const T *x,
                               T *y, const csrmv_flat_mtx_info *info) {
  VAR_FROM_CSR_DESC(T, d_csr_desc)
  const int nnz = d_csr_desc.nnz;
  const int n = d_csr_desc.cols;

  flat_set_beta_y<T, int, BEAT_Y_POLICY>(y, m, beta);

  constexpr int R = 2;
  constexpr int THREADS_PER_BLOCK = 512;

  using INDEX_TYPE = int;

  // for non one-pass.
  if (!FLAT_ONE_PASS) {
    constexpr int blocks = 512;
    constexpr int RED_OPT = FLAT_REDUCE_OPTION_VEC;
    flat_multi_pass_sparse_spmv<INDEX_TYPE, T, R, RED_OPT, 2, blocks, THREADS_PER_BLOCK>(trans, alpha, beta, m, n, nnz, rowptr,
                                                                             colindex, value, x, y);
    return;
  }

  int *break_points = info->break_points;
  if (!FLAT_ONE_PASS_ADAPTIVE) {
    constexpr int RED_OPT = FLAT_REDUCE_OPTION_VEC;
    constexpr int VEC_SIZE = 4;
    flat_one_pass_sparse_spmv<INDEX_TYPE, T, R, RED_OPT, VEC_SIZE, THREADS_PER_BLOCK>(trans, alpha, beta, m, n, nnz, rowptr,
                                                                          colindex, value, x, y, break_points);
    return;
  }

  // divide the matrix into 2 parts and get the max nnz per row of each part.
  int avg_block_nnz_max = nnz / m;
  if (avg_block_nnz_max <= 32) {
    // use single thread to reduce if average nnz of row is less than 32.
    constexpr int REDUCE_OPT = FLAT_REDUCE_OPTION_DIRECT;
    constexpr int VEC_SIZE = 1;
    flat_one_pass_sparse_spmv<INDEX_TYPE, T, R, REDUCE_OPT, VEC_SIZE, THREADS_PER_BLOCK>(trans, alpha, beta, m, n, nnz, rowptr,
                                                                             colindex, value, x, y, break_points);
  } else if (avg_block_nnz_max <= 64) {
    constexpr int REDUCE_OPT = FLAT_REDUCE_OPTION_VEC;
    // use more threads (4 threads) in vector to reduce if average nnz of row is larger than 32.
    constexpr int VEC_SIZE = 4;
    flat_one_pass_sparse_spmv<INDEX_TYPE, T, R, REDUCE_OPT, VEC_SIZE, THREADS_PER_BLOCK>(trans, alpha, beta, m, n, nnz, rowptr,
                                                                             colindex, value, x, y, break_points);
  } else {
    constexpr int REDUCE_OPT = FLAT_REDUCE_OPTION_VEC;
    // use more threads (16 threads) in vector to reduce if average nnz of row is larger than 64.
    constexpr int VEC_SIZE = 16;
    flat_one_pass_sparse_spmv<INDEX_TYPE, T, R, REDUCE_OPT, VEC_SIZE, THREADS_PER_BLOCK>(trans, alpha, beta, m, n, nnz, rowptr,
                                                                             colindex, value, x, y, break_points);
  }
}

template void adaptive_flat_sparse_spmv<double, BEAT_Y_POLICY_AUTO>(int trans, const double alpha, const double beta,
                                                                    const csr_desc<int, double> d_csr_desc,
                                                                    const double *x, double *y,
                                                                    const csrmv_flat_mtx_info *info);

template void adaptive_flat_sparse_spmv<double, BEAT_Y_POLICY_ZERO>(int trans, const double alpha, const double beta,
                                                                    const csr_desc<int, double> d_csr_desc,
                                                                    const double *x, double *y,
                                                                    const csrmv_flat_mtx_info *info);

template void adaptive_flat_sparse_spmv<double, BEAT_Y_POLICY_ONE>(int trans, const double alpha, const double beta,
                                                                   const csr_desc<int, double> d_csr_desc,
                                                                   const double *x, double *y,
                                                                   const csrmv_flat_mtx_info *info);

template void adaptive_flat_sparse_spmv<double, BEAT_Y_POLICY_OTHER>(int trans, const double alpha, const double beta,
                                                                     const csr_desc<int, double> d_csr_desc,
                                                                     const double *x, double *y,
                                                                     const csrmv_flat_mtx_info *info);

template void adaptive_flat_sparse_spmv<float, BEAT_Y_POLICY_AUTO>(int trans, const float alpha, const float beta,
                                                                   const csr_desc<int, float> d_csr_desc,
                                                                   const float *x, float *y,
                                                                   const csrmv_flat_mtx_info *info);

template void adaptive_flat_sparse_spmv<float, BEAT_Y_POLICY_ZERO>(int trans, const float alpha, const float beta,
                                                                   const csr_desc<int, float> d_csr_desc,
                                                                   const float *x, float *y,
                                                                   const csrmv_flat_mtx_info *info);

template void adaptive_flat_sparse_spmv<float, BEAT_Y_POLICY_ONE>(int trans, const float alpha, const float beta,
                                                                  const csr_desc<int, float> d_csr_desc, const float *x,
                                                                  float *y, const csrmv_flat_mtx_info *info);

template void adaptive_flat_sparse_spmv<float, BEAT_Y_POLICY_OTHER>(int trans, const float alpha, const float beta,
                                                                    const csr_desc<int, float> d_csr_desc,
                                                                    const float *x, float *y,
                                                                    const csrmv_flat_mtx_info *info);

template void flat_analyse<double>(const csr_desc<int, double> d_csr_desc, csrmv_flat_mtx_info *info);
template void flat_analyse<float>(const csr_desc<int, float> d_csr_desc, csrmv_flat_mtx_info *info);
