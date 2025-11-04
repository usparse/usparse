//
// Created by chu genshen on 2021/10/2.
//

#include "usparse_types.h"

#include "common/macros.h"
#include "line_enhance_spmv_imp.h"

template <typename INDEX_TYPE, typename FPType>
void line_enhance_sparse_spmv(int trans, const FPType alpha, const FPType beta,
                              const csr_desc<INDEX_TYPE, FPType> d_csr_desc, const FPType *x, FPType *y) {
  constexpr int R = 2;
  constexpr int ROWS_PER_BLOCK = 32; // note: make sure ROWS_PER_BLOCK * VEC_SIZE <= THREADS_PER_BLOCK.

  constexpr int REDUCE_OPTION = LE_REDUCE_OPTION_VEC;
  constexpr int VEC_SIZE = 4; // note: if using direct reduce, VEC_SIZE must set to 1.

  VAR_FROM_CSR_DESC(FPType, d_csr_desc)

  int BLOCKS = m / ROWS_PER_BLOCK + (m % ROWS_PER_BLOCK == 0 ? 0 : 1);
  constexpr int THREADS_PER_BLOCK = 512;
  LINE_ENHANCE_KERNEL_WRAPPER(FPType, REDUCE_OPTION, ROWS_PER_BLOCK, VEC_SIZE, R, BLOCKS, THREADS_PER_BLOCK);
}

template <typename INDEX_TYPE, typename FPType>
void adaptive_enhance_sparse_spmv(int trans, const FPType alpha, const FPType beta,
                                  const csr_desc<INDEX_TYPE, FPType> d_csr_desc, const FPType *x, FPType *y) {
  // common parameters:
  VAR_FROM_CSR_DESC(FPType, d_csr_desc)
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
      LINE_ENHANCE_KERNEL_WRAPPER(FPType, REDUCE_OPTION, ROWS_PER_BLOCK, VEC_SIZE, R, BLOCKS, THREADS_PER_BLOCK);
    } else { // matrix has show rows, then, use less thread(e.g. direct reduction) for reduction
      constexpr int R = 2;
      constexpr int ROWS_PER_BLOCK = 64;
      constexpr int REDUCE_OPTION = LE_REDUCE_OPTION_DIRECT;
      constexpr int VEC_SIZE = 1;

      int BLOCKS = m / ROWS_PER_BLOCK + (m % ROWS_PER_BLOCK == 0 ? 0 : 1);
      LINE_ENHANCE_KERNEL_WRAPPER(FPType, REDUCE_OPTION, ROWS_PER_BLOCK, VEC_SIZE, R, BLOCKS, THREADS_PER_BLOCK);
    }
    return;
  } else { // for large matrix
    constexpr int R = 2;
    if (nnz_per_row >= 24) { // long row matrix
      constexpr int ROWS_PER_BLOCK = 64;
      constexpr int REDUCE_OPTION = LE_REDUCE_OPTION_VEC;
      constexpr int VEC_SIZE = 4;

      int BLOCKS = m / ROWS_PER_BLOCK + (m % ROWS_PER_BLOCK == 0 ? 0 : 1);
      LINE_ENHANCE_KERNEL_WRAPPER(FPType, REDUCE_OPTION, ROWS_PER_BLOCK, VEC_SIZE, R, BLOCKS, THREADS_PER_BLOCK);
    } else {                              // short row matrix, use direct reduce
      constexpr int ROWS_PER_BLOCK = 128; // more rows it can process
      constexpr int REDUCE_OPTION = LE_REDUCE_OPTION_DIRECT;
      constexpr int VEC_SIZE = 1;

      int BLOCKS = m / ROWS_PER_BLOCK + (m % ROWS_PER_BLOCK == 0 ? 0 : 1);
      LINE_ENHANCE_KERNEL_WRAPPER(FPType, REDUCE_OPTION, ROWS_PER_BLOCK, VEC_SIZE, R, BLOCKS, THREADS_PER_BLOCK);
    }
    return;
  }
}

template void line_enhance_sparse_spmv<usparse::usparse_int, usparse::usparse_fp32>(
    int trans, const usparse::usparse_fp32 alpha, const usparse::usparse_fp32 beta,
    const csr_desc<usparse::usparse_int, usparse::usparse_fp32> d_csr_desc, const usparse::usparse_fp32 *x,
    usparse::usparse_fp32 *y);

template void line_enhance_sparse_spmv<usparse::usparse_int, usparse::usparse_fp64>(
    int trans, const usparse::usparse_fp64 alpha, const usparse::usparse_fp64 beta,
    const csr_desc<usparse::usparse_int, usparse::usparse_fp64> d_csr_desc, const usparse::usparse_fp64 *x,
    usparse::usparse_fp64 *y);

template void adaptive_enhance_sparse_spmv<usparse::usparse_int, usparse::usparse_fp64>(
    int trans, const usparse::usparse_fp64 alpha, const usparse::usparse_fp64 beta,
    const csr_desc<usparse::usparse_int, usparse::usparse_fp64> d_csr_desc, const usparse::usparse_fp64 *x,
    usparse::usparse_fp64 *y);

template void adaptive_enhance_sparse_spmv<usparse::usparse_int, usparse::usparse_fp32>(
    int trans, const usparse::usparse_fp32 alpha, const usparse::usparse_fp32 beta,
    const csr_desc<usparse::usparse_int, usparse::usparse_fp32> d_csr_desc, const usparse::usparse_fp32 *x,
    usparse::usparse_fp32 *y);
