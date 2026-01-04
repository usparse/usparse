//
// Created by genshen on 2021/7/28.
//

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "usparse_types.h"

#include "../flat/csrmv_flat.h"
#include "../line-enhance/csrmv_line_enhance.h"
#include "../line/line_strategy.h"
#include "../thread-row/thread_row.h"
#include "../vector-row/vector_row.h"

#include "common/macros.h"

template <typename INDEX_TYPE, typename DATA_TYPE>
void adaptive_sparse_spmv(int trans, const DATA_TYPE alpha, const DATA_TYPE beta,
                          const csr_desc<INDEX_TYPE, DATA_TYPE> h_csr_desc,
                          const csr_desc<INDEX_TYPE, DATA_TYPE> d_csr_desc, const DATA_TYPE *x, DATA_TYPE *y) {
  VAR_FROM_CSR_DESC(DATA_TYPE, d_csr_desc);
  const int n = d_csr_desc.cols;

  // 0. sampling and data block dividing.
  // If it need data block dividing, just use vector method,
  // but with different vector size for different data block.
  const int bp_0 = h_csr_desc.row_ptr[m / 4];
  const int bp_1 = h_csr_desc.row_ptr[m / 2];
  const int bp_2 = h_csr_desc.row_ptr[3 * m / 4];
  const int bp_3 = h_csr_desc.row_ptr[m];

  const int avg_nnz_per_row = bp_3 / m;
  const int nnz_block_0 = bp_1 - 0;
  const int nnz_block_1 = bp_3 - bp_1;

  // 1. divided into 2 data blocks if 2 data blocks have large difference.
  if ((nnz_block_1 > nnz_block_0 && nnz_block_1 / nnz_block_0 >= 4) ||
      (nnz_block_0 > nnz_block_1 && nnz_block_0 / nnz_block_1 >= 4)) {
    // vector-row based data blocks dividing
    // use nnz as weight of each block
    adaptive_vec_row_sparse_spmv(nnz_block_0, nnz_block_1, trans, alpha, beta, d_csr_desc, x, y);
    return;
  }

  // 2. otherwise, pick strategy by average nnz per row.
  if (avg_nnz_per_row <= 4) {
    // use line strategy with one-pass support.
    adaptive_line_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
    // we can also use thread-row strategy.
    // thread_row_sparse_spmv(trans, alpha, beta, m, n, row_ptr, col_index, value, x, y);
    return;
  }

  // 3. If the matrix is short row (less than 30) or is small matrix
  if (bp_3 <= 0xC00000) { // 0xC00000 = 12,582,912
    adaptive_enhance_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
    return;
  }

  // 4. If non-zeros number of large enough, we use flat strategy.
  // The flat strategy has a data pre-processing kernel function,
  // which can not be applied to small data set.
  if (bp_3 > (1 << 23)) { // 2^23 = 8,388,608
    csrmv_flat_mtx_info info;
    auto d_csr_mat_descr = usparse::csr_mat_descr<DATA_TYPE, INDEX_TYPE, INDEX_TYPE, INDEX_TYPE>(d_csr_desc);
    flat_analyse<DATA_TYPE, INDEX_TYPE, INDEX_TYPE, INDEX_TYPE>(d_csr_mat_descr, &info);
    adaptive_flat_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y, &info);
    return;
  }

  // 5. use vector-row method for small data set.
  line_enhance_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
}

template void adaptive_sparse_spmv<usparse::usparse_int, usparse::usparse_fp32>(
    int trans, const usparse::usparse_fp32 alpha, const usparse::usparse_fp32 beta,
    const csr_desc<usparse::usparse_int, usparse::usparse_fp32> h_csr_desc,
    const csr_desc<usparse::usparse_int, usparse::usparse_fp32> d_csr_desc, const usparse::usparse_fp32 *x,
    usparse::usparse_fp32 *y);

template void adaptive_sparse_spmv<usparse::usparse_int, usparse::usparse_fp64>(
    int trans, const usparse::usparse_fp64 alpha, const usparse::usparse_fp64 beta,
    const csr_desc<usparse::usparse_int, usparse::usparse_fp64> h_csr_desc,
    const csr_desc<usparse::usparse_int, usparse::usparse_fp64> d_csr_desc, const usparse::usparse_fp64 *x,
    usparse::usparse_fp64 *y);
