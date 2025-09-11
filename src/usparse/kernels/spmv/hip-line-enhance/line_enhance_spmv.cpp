//
// Created by chu genshen on 2021/10/2.
//

#include "../api/types.h"
#include "../common/CPU/csr_macros.h"
#include "line_enhance_spmv_imp.h"

template <int BEAT_Y_POLICY>
void adaptive_enhance_sparse_spmv_fp32(int trans, const float alpha, const float beta,
                                       const csr_desc<int, float> d_csr_desc, const float *x, float *y);

template <int BEAT_Y_POLICY>
void adaptive_enhance_sparse_spmv_fp64(int trans, const double alpha, const double beta,
                                       const csr_desc<int, double> d_csr_desc, const double *x, double *y);

template <typename FPType>
void line_enhance_sparse_spmv(int trans, const FPType alpha, const FPType beta, const csr_desc<int, FPType> d_csr_desc,
                              const FPType *x, FPType *y) {
  constexpr int R = 2;
  constexpr int ROWS_PER_BLOCK = 32; // note: make sure ROWS_PER_BLOCK * VEC_SIZE <= THREADS_PER_BLOCK.

  constexpr int REDUCE_OPTION = LE_REDUCE_OPTION_VEC;
  constexpr int VEC_SIZE = 4; // note: if using direct reduce, VEC_SIZE must set to 1.

  VAR_FROM_CSR_DESC(FPType, d_csr_desc)

  int BLOCKS = m / ROWS_PER_BLOCK + (m % ROWS_PER_BLOCK == 0 ? 0 : 1);
  constexpr int THREADS_PER_BLOCK = 512;
  // todo: replace BEAT_Y_POLICY_AUTO with template paramter
  LINE_ENHANCE_KERNEL_WRAPPER(FPType, BEAT_Y_POLICY_AUTO, REDUCE_OPTION, ROWS_PER_BLOCK, VEC_SIZE, R, BLOCKS,
                              THREADS_PER_BLOCK);
}

template <typename FPType, int BEAT_Y_POLICY>
void adaptive_enhance_sparse_spmv(int trans, const FPType alpha, const FPType beta,
                                  const csr_desc<int, FPType> d_csr_desc, const FPType *x, FPType *y) {
  VAR_FROM_CSR_DESC(FPType, d_csr_desc)
  const int nnz = d_csr_desc.nnz;
  const int n = d_csr_desc.cols;

  if constexpr (std::is_same<FPType, double>::value) { // float64
    adaptive_enhance_sparse_spmv_fp64<BEAT_Y_POLICY>(trans, alpha, beta, d_csr_desc, x, y);
  } else { // float32
    adaptive_enhance_sparse_spmv_fp32<BEAT_Y_POLICY>(trans, alpha, beta, d_csr_desc, x, y);
  }
  return;

#if 0
  // common parameters:
  constexpr int THREADS_PER_BLOCK = 512;

  // for small matrix.
  const int mtx_nnz = d_csr_desc.nnz;
  const int nnz_per_row = mtx_nnz / m;
  if (mtx_nnz <= (1 << 24)) { // 2^24=16,777,216
    if (nnz_per_row >= 32) {  // matrix has long rows
      constexpr int R = 4;
      constexpr int ROWS_PER_BLOCK = 64;
      constexpr int REDUCE_OPTION = LE_REDUCE_OPTION_VEC;
      constexpr int VEC_SIZE = 8;

      int BLOCKS = m / ROWS_PER_BLOCK + (m % ROWS_PER_BLOCK == 0 ? 0 : 1);
      LINE_ENHANCE_KERNEL_WRAPPER(FPType, BEAT_Y_POLICY, REDUCE_OPTION, ROWS_PER_BLOCK, VEC_SIZE, R, BLOCKS,
                                  THREADS_PER_BLOCK);
    } else { // matrix has show rows, then, use less thread(e.g. direct reduction) for reduction
      constexpr int R = 2;
      constexpr int ROWS_PER_BLOCK = 64;
      constexpr int REDUCE_OPTION = LE_REDUCE_OPTION_DIRECT;
      constexpr int VEC_SIZE = 1;

      int BLOCKS = m / ROWS_PER_BLOCK + (m % ROWS_PER_BLOCK == 0 ? 0 : 1);
      LINE_ENHANCE_KERNEL_WRAPPER(FPType, BEAT_Y_POLICY, REDUCE_OPTION, ROWS_PER_BLOCK, VEC_SIZE, R, BLOCKS,
                                  THREADS_PER_BLOCK);
    }
    return;
  } else { // for large matrix
    constexpr int R = 2;
    if (nnz_per_row >= 24) { // long row matrix
      constexpr int ROWS_PER_BLOCK = 64;
      constexpr int REDUCE_OPTION = LE_REDUCE_OPTION_VEC;
      constexpr int VEC_SIZE = 4;

      int BLOCKS = m / ROWS_PER_BLOCK + (m % ROWS_PER_BLOCK == 0 ? 0 : 1);
      LINE_ENHANCE_KERNEL_WRAPPER(FPType, BEAT_Y_POLICY, REDUCE_OPTION, ROWS_PER_BLOCK, VEC_SIZE, R, BLOCKS,
                                  THREADS_PER_BLOCK);
    } else {                              // short row matrix, use direct reduce
      constexpr int ROWS_PER_BLOCK = 128; // more rows it can process
      constexpr int REDUCE_OPTION = LE_REDUCE_OPTION_DIRECT;
      constexpr int VEC_SIZE = 1;

      int BLOCKS = m / ROWS_PER_BLOCK + (m % ROWS_PER_BLOCK == 0 ? 0 : 1);
      LINE_ENHANCE_KERNEL_WRAPPER(FPType, BEAT_Y_POLICY, REDUCE_OPTION, ROWS_PER_BLOCK, VEC_SIZE, R, BLOCKS,
                                  THREADS_PER_BLOCK);
    }
    return;
  }
#endif
}

template void adaptive_enhance_sparse_spmv<double, BEAT_Y_POLICY_AUTO>(int trans, const double alpha, const double beta,
                                                                       const csr_desc<int, double> d_csr_desc,
                                                                       const double *x, double *y);

template void adaptive_enhance_sparse_spmv<float, BEAT_Y_POLICY_AUTO>(int trans, const float alpha, const float beta,
                                                                      const csr_desc<int, float> d_csr_desc,
                                                                      const float *x, float *y);

template void adaptive_enhance_sparse_spmv<double, BEAT_Y_POLICY_ZERO>(int trans, const double alpha, const double beta,
                                                                       const csr_desc<int, double> d_csr_desc,
                                                                       const double *x, double *y);

template void adaptive_enhance_sparse_spmv<float, BEAT_Y_POLICY_ZERO>(int trans, const float alpha, const float beta,
                                                                      const csr_desc<int, float> d_csr_desc,
                                                                      const float *x, float *y);

template void adaptive_enhance_sparse_spmv<double, BEAT_Y_POLICY_ONE>(int trans, const double alpha, const double beta,
                                                                      const csr_desc<int, double> d_csr_desc,
                                                                      const double *x, double *y);

template void adaptive_enhance_sparse_spmv<float, BEAT_Y_POLICY_ONE>(int trans, const float alpha, const float beta,
                                                                     const csr_desc<int, float> d_csr_desc,
                                                                     const float *x, float *y);

template void adaptive_enhance_sparse_spmv<double, BEAT_Y_POLICY_OTHER>(int trans, const double alpha,
                                                                        const double beta,
                                                                        const csr_desc<int, double> d_csr_desc,
                                                                        const double *x, double *y);

template void adaptive_enhance_sparse_spmv<float, BEAT_Y_POLICY_OTHER>(int trans, const float alpha, const float beta,
                                                                       const csr_desc<int, float> d_csr_desc,
                                                                       const float *x, float *y);

template <int BEAT_Y_POLICY>
void adaptive_enhance_sparse_spmv_fp64(int trans, const double alpha, const double beta,
                                       const csr_desc<int, double> d_csr_desc, const double *x, double *y) {
  using T = double;
  VAR_FROM_CSR_DESC(T, d_csr_desc)
  const int nnz = d_csr_desc.nnz;
  const int n = d_csr_desc.cols;

  // common parameters:
  constexpr int ROWS_PER_BLOCK = 64;
  const int density = nnz / n;
  if (density < 10) {
    constexpr int R = 2;
    constexpr int THREADS_PER_BLOCK = 256;
    constexpr int REDUCE_OPTION = LE_REDUCE_OPTION_DIRECT;
    constexpr int VEC_SIZE = 1;
    int BLOCKS = m / ROWS_PER_BLOCK + (m % ROWS_PER_BLOCK == 0 ? 0 : 1);
    LINE_ENHANCE_KERNEL_WRAPPER(T, BEAT_Y_POLICY, REDUCE_OPTION, ROWS_PER_BLOCK, VEC_SIZE, R, BLOCKS, THREADS_PER_BLOCK);
  } else if (density < 50) {
    constexpr int R = 4;
    constexpr int THREADS_PER_BLOCK = 512;
    constexpr int REDUCE_OPTION = LE_REDUCE_OPTION_VEC;
    constexpr int VEC_SIZE = 4;
    int BLOCKS = m / ROWS_PER_BLOCK + (m % ROWS_PER_BLOCK == 0 ? 0 : 1);
    LINE_ENHANCE_KERNEL_WRAPPER(T, BEAT_Y_POLICY, REDUCE_OPTION, ROWS_PER_BLOCK, VEC_SIZE, R, BLOCKS, THREADS_PER_BLOCK);
  } else {
    constexpr int R = 4;
    constexpr int THREADS_PER_BLOCK = 512;
    constexpr int REDUCE_OPTION = LE_REDUCE_OPTION_VEC;
    constexpr int VEC_SIZE = 8;
    int BLOCKS = m / ROWS_PER_BLOCK + (m % ROWS_PER_BLOCK == 0 ? 0 : 1);
    LINE_ENHANCE_KERNEL_WRAPPER(T, BEAT_Y_POLICY, REDUCE_OPTION, ROWS_PER_BLOCK, VEC_SIZE, R, BLOCKS, THREADS_PER_BLOCK);
  }
}

template <int BEAT_Y_POLICY>
void adaptive_enhance_sparse_spmv_fp32(int trans, const float alpha, const float beta,
                                       const csr_desc<int, float> d_csr_desc, const float *x, float *y) {
  using T = float;
  VAR_FROM_CSR_DESC(T, d_csr_desc)
  const int nnz = d_csr_desc.nnz;
  const int n = d_csr_desc.cols;

  // common parameters:
  constexpr int ROWS_PER_BLOCK = 64;
  const int density = nnz / n;
  if (density < 10) {
    constexpr int R = 2;
    constexpr int THREADS_PER_BLOCK = 256;
    constexpr int REDUCE_OPTION = LE_REDUCE_OPTION_DIRECT;
    constexpr int VEC_SIZE = 1;
    int BLOCKS = m / ROWS_PER_BLOCK + (m % ROWS_PER_BLOCK == 0 ? 0 : 1);
    LINE_ENHANCE_KERNEL_WRAPPER(T, BEAT_Y_POLICY, REDUCE_OPTION, ROWS_PER_BLOCK, VEC_SIZE, R, BLOCKS, THREADS_PER_BLOCK);
  } else if (density < 50) {
    constexpr int R = 4;
    constexpr int THREADS_PER_BLOCK = 512;
    constexpr int REDUCE_OPTION = LE_REDUCE_OPTION_VEC;
    constexpr int VEC_SIZE = 4;
    int BLOCKS = m / ROWS_PER_BLOCK + (m % ROWS_PER_BLOCK == 0 ? 0 : 1);
    LINE_ENHANCE_KERNEL_WRAPPER(T, BEAT_Y_POLICY, REDUCE_OPTION, ROWS_PER_BLOCK, VEC_SIZE, R, BLOCKS, THREADS_PER_BLOCK);
  } else {
    constexpr int R = 4;
    constexpr int THREADS_PER_BLOCK = 512;
    constexpr int REDUCE_OPTION = LE_REDUCE_OPTION_VEC;
    constexpr int VEC_SIZE = 8;
    int BLOCKS = m / ROWS_PER_BLOCK + (m % ROWS_PER_BLOCK == 0 ? 0 : 1);
    LINE_ENHANCE_KERNEL_WRAPPER(T, BEAT_Y_POLICY, REDUCE_OPTION, ROWS_PER_BLOCK, VEC_SIZE, R, BLOCKS, THREADS_PER_BLOCK);
  }
}
