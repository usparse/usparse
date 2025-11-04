//
// Created by genshen on 2021/7/15.
//

#include "usparse_types.h"

#include "common/macros.h"
#include "line_strategy.h"

template <typename INDEX_TYPE, typename DATA_TYPE>
void line_sparse_spmv(int trans, const DATA_TYPE alpha, const DATA_TYPE beta,
                      const csr_desc<INDEX_TYPE, DATA_TYPE> d_csr_desc, const DATA_TYPE *x, DATA_TYPE *y) {
  VAR_FROM_CSR_DESC(DATA_TYPE, d_csr_desc)

  // const int avg_eles_per_row = ceil(d_csr_desc.nnz + 0.0 / m);
  const int avg_eles_per_row = d_csr_desc.nnz / m;
  // ROW_NUM * MAX_NNZ_NUM < HIP_THREAD
  constexpr int HIP_THREAD = 256;
  constexpr int R = 2;
  constexpr int BLOCK_LDS_SIZE = HIP_THREAD * R;

#define LINE_ONE_PASS_KERNEL_WRAPPER(N, BLOCK_LDS_SIZE, BLOCKS, THREADS)                                               \
  (spmv_line_one_pass_kernel<N, BLOCK_LDS_SIZE, INDEX_TYPE, DATA_TYPE>)<<<BLOCKS, THREADS>>>(m, alpha, beta, rowptr,   \
                                                                                             colindex, value, x, y)

  if (avg_eles_per_row <= 5) {
    constexpr int MAX_NNZ_NUM = 5;
    const int ROW_NUM = HIP_THREAD / MAX_NNZ_NUM * R;
    const int HIP_BLOCK = m / ROW_NUM + (m % ROW_NUM == 0 ? 0 : 1);
    LINE_ONE_PASS_KERNEL_WRAPPER(ROW_NUM, BLOCK_LDS_SIZE, HIP_BLOCK, HIP_THREAD);
  } else {
    constexpr int MAX_NNZ_NUM = 14;
    constexpr int ROW_NUM = HIP_THREAD / MAX_NNZ_NUM * R;
    const int HIP_BLOCK = m / ROW_NUM + (m % ROW_NUM == 0 ? 0 : 1);
    LINE_ONE_PASS_KERNEL_WRAPPER(ROW_NUM, BLOCK_LDS_SIZE, HIP_BLOCK, HIP_THREAD);
  }
// LINE_KERNEL_WRAPPER(5);
#undef LINE_ONE_PASS_KERNEL_WRAPPER
}

/**
 * @tparam NNZ_PER_ROW NNZ_PER_ROW is an estimated value of the max nnz of matrix rows.
 *  suggested value is (2 * VEC_SIZE + 1).
 */
template <int BLOCK_LDS_SIZE, int VEC_SIZE, int HIP_THREADS, typename I, typename T>
void inline adaptive_line_wrapper(const int NNZ_PER_ROW, const I m, const T alpha, const T beta, const I *row_offset,
                                  const I *csr_col_ind, const T *csr_val, const T *x, T *y) {
  const int ROW_NUM = BLOCK_LDS_SIZE / NNZ_PER_ROW;

  const int HIP_BLOCKS = m / ROW_NUM + (m % ROW_NUM == 0 ? 0 : 1);
  (spmv_adaptive_line_kernel<BLOCK_LDS_SIZE, HIP_THREADS, __WF_SIZE__, VEC_SIZE, I, T,
                             false>)<<<HIP_BLOCKS, HIP_THREADS>>>(ROW_NUM, m, alpha, beta, row_offset, csr_col_ind,
                                                                  csr_val, x, y);
}

#define ADAPTIVE_LINE_WRAPPER(VEC_SIZE, NNZ_PER_ROW)                                                                   \
  (adaptive_line_wrapper<BLOCK_LDS_SIZE, VEC_SIZE, HIP_THREADS, INDEX_TYPE, DATA_TYPE>)((NNZ_PER_ROW), m, alpha, beta, \
                                                                                        rowptr, colindex, value, x, y)

template <typename INDEX_TYPE, typename DATA_TYPE>
void adaptive_line_sparse_spmv(int trans, const DATA_TYPE alpha, const DATA_TYPE beta,
                               const csr_desc<INDEX_TYPE, DATA_TYPE> d_csr_desc, const DATA_TYPE *x, DATA_TYPE *y) {
  VAR_FROM_CSR_DESC(DATA_TYPE, d_csr_desc)

  const int nnz_per_row = d_csr_desc.nnz / m;
  // make ROW_NUM * NNZ_PER_ROW <= HIP_THREAD
  constexpr int HIP_THREADS = 256;
  constexpr int R = 2;
  constexpr int BLOCK_LDS_SIZE = HIP_THREADS * R;

  if (nnz_per_row <= 4 || __WF_SIZE__ <= 2) {
    // by setting NNZ_PER_ROW's value (which can change and can only change ROW_NUM),
    // we can try to make rows fall into line algorithm, rather than vector row.
    ADAPTIVE_LINE_WRAPPER(2, nnz_per_row + 1);
  } else if (nnz_per_row <= 8 || __WF_SIZE__ <= 4) {
    ADAPTIVE_LINE_WRAPPER(4, nnz_per_row + 1);
  } else if (nnz_per_row <= 16 || __WF_SIZE__ <= 8) {
    ADAPTIVE_LINE_WRAPPER(8, nnz_per_row + 2);
  } else if (nnz_per_row <= 32 || __WF_SIZE__ <= 16) {
    ADAPTIVE_LINE_WRAPPER(16, nnz_per_row + 4);
  } else if (nnz_per_row <= 64 || __WF_SIZE__ <= 32) {
    ADAPTIVE_LINE_WRAPPER(32, nnz_per_row + 4);
  } else {
    ADAPTIVE_LINE_WRAPPER(64, nnz_per_row + 4);
  }
}

template void line_sparse_spmv<usparse::usparse_int, usparse::usparse_fp32>(
    int trans, const usparse::usparse_fp32 alpha, const usparse::usparse_fp32 beta,
    const csr_desc<usparse::usparse_int, usparse::usparse_fp32> d_csr_desc, const usparse::usparse_fp32 *d_x,
    usparse::usparse_fp32 *d_y);

template void line_sparse_spmv<usparse::usparse_int, usparse::usparse_fp64>(
    int trans, const usparse::usparse_fp64 alpha, const usparse::usparse_fp64 beta,
    const csr_desc<usparse::usparse_int, usparse::usparse_fp64> d_csr_desc, const usparse::usparse_fp64 *d_x,
    usparse::usparse_fp64 *d_y);

template void adaptive_line_sparse_spmv<usparse::usparse_int, usparse::usparse_fp32>(
    int trans, const usparse::usparse_fp32 alpha, const usparse::usparse_fp32 beta,
    const csr_desc<usparse::usparse_int, usparse::usparse_fp32> d_csr_desc, const usparse::usparse_fp32 *d_x,
    usparse::usparse_fp32 *d_y);

template void adaptive_line_sparse_spmv<usparse::usparse_int, usparse::usparse_fp64>(
    int trans, const usparse::usparse_fp64 alpha, const usparse::usparse_fp64 beta,
    const csr_desc<usparse::usparse_int, usparse::usparse_fp64> d_csr_desc, const usparse::usparse_fp64 *d_x,
    usparse::usparse_fp64 *d_y);
