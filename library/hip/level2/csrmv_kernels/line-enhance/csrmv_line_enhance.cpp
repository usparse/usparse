//
// Created by chu genshen on 2021/10/2.
//

#include "common/macros.h"
#include "line_enhance_kernels_imp.h"
#include "usparse_types.h"

template <typename COMPUTE_TYPE, typename NNZ_TYPE, typename ROW_TYPE, typename COL_TYPE, typename SCALE, typename A,
          typename X, typename Y, int R, int ROWS_PER_BLOCK, int VEC_SIZE, int THREADS_PER_BLOCK>
void line_enhance_sparse_spmv(hipStream_t stream, int trans, const SCALE alpha, const SCALE beta,
                              const usparse::csr_desc<NNZ_TYPE, A, ROW_TYPE, COL_TYPE> d_csr_desc, const X *x, Y *y) {
  EXPAND_FROM_CSR_MAT_DESCR(A, NNZ_TYPE, ROW_TYPE, COL_TYPE, d_csr_desc)

  constexpr int REDUCE_OPTION = VEC_SIZE == 0 ? LE_REDUCE_OPTION_DIRECT : LE_REDUCE_OPTION_VEC;

  int BLOCKS = m / ROWS_PER_BLOCK + (m % ROWS_PER_BLOCK == 0 ? 0 : 1);
  LINE_ENHANCE_KERNEL_WRAPPER(stream, COMPUTE_TYPE, NNZ_TYPE, ROW_TYPE, A, X, Y, REDUCE_OPTION, ROWS_PER_BLOCK,
                              VEC_SIZE, R, BLOCKS, THREADS_PER_BLOCK);
}

template <typename INDEX_TYPE, typename DATA_TYPE>
void line_enhance_sparse_spmv(int trans, const DATA_TYPE alpha, const DATA_TYPE beta,
                              const usparse::csr_desc<INDEX_TYPE, DATA_TYPE, INDEX_TYPE, INDEX_TYPE> d_csr_desc,
                              const DATA_TYPE *x, DATA_TYPE *y) {
  constexpr int R = 2;
  constexpr int ROWS_PER_BLOCK = 32; // note: make sure ROWS_PER_BLOCK * VEC_SIZE <= THREADS_PER_BLOCK.
  constexpr int VEC_SIZE = 4;        // note: if using direct reduce, VEC_SIZE must set to 1.
  constexpr int THREADS_PER_BLOCK = 512;

  hipStream_t stream = 0; // default stream
  line_enhance_sparse_spmv<DATA_TYPE, INDEX_TYPE, INDEX_TYPE, INDEX_TYPE, DATA_TYPE, DATA_TYPE, DATA_TYPE, DATA_TYPE, R,
                           ROWS_PER_BLOCK, VEC_SIZE, THREADS_PER_BLOCK>(stream, trans, alpha, beta, d_csr_desc, x, y);
}

// todo: better MACRO define
#define USPARSE_ENABLE_TUNER
#ifdef USPARSE_ENABLE_TUNER
// todo: cpp17 checking
template <typename COMPUTE_TYPE, typename NNZ_TYPE, typename ROW_TYPE, typename COL_TYPE, typename SCALE_TYPE,
          typename A, typename X, typename Y>
void __not_called_line_enhance_template_instantiate() {
  constexpr std::array<int, 4> R = {1, 2, 3, 4};
  constexpr std::array<int, 4> ROWS_PER_BLOCK = {16, 32, 64, 128};
  constexpr std::array<int, 7> VEC_SIZE = {1, 2, 4, 8, 16, 32, 64};
  constexpr std::array<int, 2> THREADS_PER_BLOCK = {256, 512};

  usparse::csr_desc<NNZ_TYPE, A, ROW_TYPE, COL_TYPE> d_csr_desc(0, 0, 0, nullptr, nullptr, nullptr);

  constexpr size_t S1 = R.size();
  constexpr size_t S2 = ROWS_PER_BLOCK.size();
  constexpr size_t S3 = VEC_SIZE.size();
  constexpr size_t S4 = THREADS_PER_BLOCK.size();

  [&]<size_t... TotalIdx>(std::index_sequence<TotalIdx...>) {
    (([&] {
       // get index
       constexpr size_t i4 = TotalIdx % S4;
       constexpr size_t i3 = (TotalIdx / S4) % S3;
       constexpr size_t i2 = (TotalIdx / (S4 * S3)) % S2;
       constexpr size_t i1 = (TotalIdx / (S4 * S3 * S2)) % S1;

       constexpr int _r = R[i1];
       constexpr int _rows_per_block = ROWS_PER_BLOCK[i2];
       constexpr int _vec_size = VEC_SIZE[i3];
       constexpr int _threads_per_block = THREADS_PER_BLOCK[i4];
       if constexpr (_rows_per_block * _vec_size <= _threads_per_block) {
         line_enhance_sparse_spmv<COMPUTE_TYPE, NNZ_TYPE, ROW_TYPE, COL_TYPE, SCALE_TYPE, A, X, Y, _r, _rows_per_block,
                                  _vec_size, _threads_per_block>(0, 0, 0.0, 0.0, d_csr_desc, nullptr, nullptr);
       }
     }()),
     ...);
  }(std::make_index_sequence<S1 * S2 * S3 * S4>{});
}

template void
__not_called_line_enhance_template_instantiate<usparse::usparse_fp64, usparse::usparse_int, usparse::usparse_int,
                                               usparse::usparse_int, usparse::usparse_fp64, usparse::usparse_fp64,
                                               usparse::usparse_fp64, usparse::usparse_fp64>();

template void
__not_called_line_enhance_template_instantiate<usparse::usparse_fp32, usparse::usparse_int, usparse::usparse_int,
                                               usparse::usparse_int, usparse::usparse_fp32, usparse::usparse_fp32,
                                               usparse::usparse_fp32, usparse::usparse_fp32>();

template void
__not_called_line_enhance_template_instantiate<usparse::usparse_fpc64, usparse::usparse_int, usparse::usparse_int,
                                               usparse::usparse_int, usparse::usparse_fpc64, usparse::usparse_fpc64,
                                               usparse::usparse_fpc64, usparse::usparse_fpc64>();

template void
__not_called_line_enhance_template_instantiate<usparse::usparse_fpc32, usparse::usparse_int, usparse::usparse_int,
                                               usparse::usparse_int, usparse::usparse_fpc32, usparse::usparse_fpc32,
                                               usparse::usparse_fpc32, usparse::usparse_fpc32>();

#endif

template <typename COMPUTE, typename NNZ, typename ROW, typename COL, typename SCALE, typename A, typename X,
          typename Y>
void adaptive_enhance_sparse_spmv(hipStream_t stream, int trans, const SCALE alpha, const SCALE beta,
                                  const usparse::csr_desc<NNZ, A, ROW, COL> d_csr_desc, const X *x, Y *y) {
  // common parameters:
  EXPAND_FROM_CSR_MAT_DESCR(A, NNZ, ROW, COL, d_csr_desc)

  constexpr int THREADS_PER_BLOCK = 512;

  // for small matrix.
  const NNZ mtx_nnz = d_csr_desc.nnz;
  const NNZ nnz_per_row = mtx_nnz / m;
  if (mtx_nnz <= (1 << 24)) { // 2^24=16,777,216
    if (nnz_per_row >= 32) {  // matrix has long rows
      constexpr int R = 4;
      constexpr int ROWS_PER_BLOCK = 64;
      constexpr int REDUCE_OPTION = LE_REDUCE_OPTION_VEC;
      constexpr int VEC_SIZE = 8;

      int BLOCKS = m / ROWS_PER_BLOCK + (m % ROWS_PER_BLOCK == 0 ? 0 : 1);
      LINE_ENHANCE_KERNEL_WRAPPER(stream, COMPUTE, NNZ, ROW, A, X, Y, REDUCE_OPTION, ROWS_PER_BLOCK, VEC_SIZE, R,
                                  BLOCKS, THREADS_PER_BLOCK);
    } else { // matrix has show rows, then, use less thread(e.g. direct reduction) for reduction
      constexpr int R = 2;
      constexpr int ROWS_PER_BLOCK = 64;
      constexpr int REDUCE_OPTION = LE_REDUCE_OPTION_DIRECT;
      constexpr int VEC_SIZE = 1;

      int BLOCKS = m / ROWS_PER_BLOCK + (m % ROWS_PER_BLOCK == 0 ? 0 : 1);
      LINE_ENHANCE_KERNEL_WRAPPER(stream, COMPUTE, NNZ, ROW, A, X, Y, REDUCE_OPTION, ROWS_PER_BLOCK, VEC_SIZE, R,
                                  BLOCKS, THREADS_PER_BLOCK);
    }
    return;
  } else { // for large matrix
    constexpr int R = 2;
    if (nnz_per_row >= 24) { // long row matrix
      constexpr int ROWS_PER_BLOCK = 64;
      constexpr int REDUCE_OPTION = LE_REDUCE_OPTION_VEC;
      constexpr int VEC_SIZE = 4;

      int BLOCKS = m / ROWS_PER_BLOCK + (m % ROWS_PER_BLOCK == 0 ? 0 : 1);
      LINE_ENHANCE_KERNEL_WRAPPER(stream, COMPUTE, NNZ, ROW, A, X, Y, REDUCE_OPTION, ROWS_PER_BLOCK, VEC_SIZE, R,
                                  BLOCKS, THREADS_PER_BLOCK);
    } else {                              // short row matrix, use direct reduce
      constexpr int ROWS_PER_BLOCK = 128; // more rows it can process
      constexpr int REDUCE_OPTION = LE_REDUCE_OPTION_DIRECT;
      constexpr int VEC_SIZE = 1;

      int BLOCKS = m / ROWS_PER_BLOCK + (m % ROWS_PER_BLOCK == 0 ? 0 : 1);
      LINE_ENHANCE_KERNEL_WRAPPER(stream, COMPUTE, NNZ, ROW, A, X, Y, REDUCE_OPTION, ROWS_PER_BLOCK, VEC_SIZE, R,
                                  BLOCKS, THREADS_PER_BLOCK);
    }
    return;
  }
}

template <typename COMPUTE, typename NNZ, typename ROW, typename COL, typename SCALE, typename A, typename X,
          typename Y>
void adaptive_enhance_sparse_spmv(int trans, const SCALE alpha, const SCALE beta,
                                  const usparse::csr_desc<NNZ, A, ROW, COL> d_csr_desc, const X *x, Y *y) {
  hipStream_t stream = 0; // default stream
  adaptive_enhance_sparse_spmv<COMPUTE, NNZ, ROW, COL, SCALE, A, X, Y>(stream, trans, alpha, beta, d_csr_desc, x, y);
}

#define LINE_ENHANCE_INSTANTIATE(DATA_TYPE, NNZ_TYPE, ROW_COL_TYPE)                                                    \
  template void adaptive_enhance_sparse_spmv<DATA_TYPE, NNZ_TYPE, ROW_COL_TYPE, ROW_COL_TYPE, DATA_TYPE, DATA_TYPE,    \
                                             DATA_TYPE, DATA_TYPE>(                                                    \
      hipStream_t stream, int trans, const DATA_TYPE alpha, const DATA_TYPE beta,                                      \
      const usparse::csr_desc<NNZ_TYPE, DATA_TYPE, ROW_COL_TYPE, ROW_COL_TYPE> d_csr_desc, const DATA_TYPE *x,         \
      DATA_TYPE *y);                                                                                                   \
  template void adaptive_enhance_sparse_spmv<DATA_TYPE, NNZ_TYPE, ROW_COL_TYPE, ROW_COL_TYPE, DATA_TYPE, DATA_TYPE,    \
                                             DATA_TYPE, DATA_TYPE>(                                                    \
      int trans, const DATA_TYPE alpha, const DATA_TYPE beta,                                                          \
      const usparse::csr_desc<NNZ_TYPE, DATA_TYPE, ROW_COL_TYPE, ROW_COL_TYPE> d_csr_desc, const DATA_TYPE *x,         \
      DATA_TYPE *y);

LINE_ENHANCE_INSTANTIATE(usparse::usparse_fp32, int32_t, int32_t);
LINE_ENHANCE_INSTANTIATE(usparse::usparse_fp32, int64_t, int32_t);
LINE_ENHANCE_INSTANTIATE(usparse::usparse_fp32, int64_t, int64_t);
LINE_ENHANCE_INSTANTIATE(usparse::usparse_fp64, int32_t, int32_t);
LINE_ENHANCE_INSTANTIATE(usparse::usparse_fp64, int64_t, int32_t);
LINE_ENHANCE_INSTANTIATE(usparse::usparse_fp64, int64_t, int64_t);

LINE_ENHANCE_INSTANTIATE(usparse::usparse_fpc32, int32_t, int32_t);
LINE_ENHANCE_INSTANTIATE(usparse::usparse_fpc32, int64_t, int32_t);
LINE_ENHANCE_INSTANTIATE(usparse::usparse_fpc32, int64_t, int64_t);
LINE_ENHANCE_INSTANTIATE(usparse::usparse_fpc64, int32_t, int32_t);
LINE_ENHANCE_INSTANTIATE(usparse::usparse_fpc64, int64_t, int32_t);
LINE_ENHANCE_INSTANTIATE(usparse::usparse_fpc64, int64_t, int64_t);

#define LINE_ENHANCE_INSTANTIATE_MIXED(TTYPE, NNZ_TYPE, ROW_COL_TYPE, ATYPE, XTYPE, YTYPE)                             \
  template void adaptive_enhance_sparse_spmv<TTYPE, NNZ_TYPE, ROW_COL_TYPE, ROW_COL_TYPE, TTYPE, ATYPE, XTYPE, YTYPE>( \
      hipStream_t stream, int trans, const TTYPE alpha, const TTYPE beta,                                              \
      const usparse::csr_desc<NNZ_TYPE, ATYPE, ROW_COL_TYPE, ROW_COL_TYPE> d_csr_desc, const XTYPE *x, YTYPE *y);      \
  template void adaptive_enhance_sparse_spmv<TTYPE, NNZ_TYPE, ROW_COL_TYPE, ROW_COL_TYPE, TTYPE, ATYPE, XTYPE, YTYPE>( \
      int trans, const TTYPE alpha, const TTYPE beta,                                                                  \
      const usparse::csr_desc<NNZ_TYPE, ATYPE, ROW_COL_TYPE, ROW_COL_TYPE> d_csr_desc, const XTYPE *x, YTYPE *y);

LINE_ENHANCE_INSTANTIATE_MIXED(int32_t, int32_t, int32_t, int8_t, int8_t, int32_t)
LINE_ENHANCE_INSTANTIATE_MIXED(int32_t, int64_t, int32_t, int8_t, int8_t, int32_t)
LINE_ENHANCE_INSTANTIATE_MIXED(int32_t, int64_t, int64_t, int8_t, int8_t, int32_t)
LINE_ENHANCE_INSTANTIATE_MIXED(float, int32_t, int32_t, int8_t, int8_t, float)
LINE_ENHANCE_INSTANTIATE_MIXED(float, int64_t, int32_t, int8_t, int8_t, float)
LINE_ENHANCE_INSTANTIATE_MIXED(float, int64_t, int64_t, int8_t, int8_t, float)
LINE_ENHANCE_INSTANTIATE_MIXED(usparse::usparse_fpc32, int32_t, int32_t, float, usparse::usparse_fpc32,
                               usparse::usparse_fpc32)
LINE_ENHANCE_INSTANTIATE_MIXED(usparse::usparse_fpc32, int64_t, int32_t, float, usparse::usparse_fpc32,
                               usparse::usparse_fpc32)
LINE_ENHANCE_INSTANTIATE_MIXED(usparse::usparse_fpc32, int64_t, int64_t, float, usparse::usparse_fpc32,
                               usparse::usparse_fpc32)

LINE_ENHANCE_INSTANTIATE_MIXED(double, int32_t, int32_t, float, double, double)
LINE_ENHANCE_INSTANTIATE_MIXED(double, int64_t, int32_t, float, double, double)
LINE_ENHANCE_INSTANTIATE_MIXED(double, int64_t, int64_t, float, double, double)

LINE_ENHANCE_INSTANTIATE_MIXED(usparse::usparse_fpc64, int32_t, int32_t, double, usparse::usparse_fpc64,
                               usparse::usparse_fpc64)
LINE_ENHANCE_INSTANTIATE_MIXED(usparse::usparse_fpc64, int64_t, int32_t, double, usparse::usparse_fpc64,
                               usparse::usparse_fpc64)
LINE_ENHANCE_INSTANTIATE_MIXED(usparse::usparse_fpc64, int64_t, int64_t, double, usparse::usparse_fpc64,
                               usparse::usparse_fpc64)

LINE_ENHANCE_INSTANTIATE_MIXED(usparse::usparse_fpc64, int32_t, int32_t, usparse::usparse_fpc32, usparse::usparse_fpc64,
                               usparse::usparse_fpc64)
LINE_ENHANCE_INSTANTIATE_MIXED(usparse::usparse_fpc64, int64_t, int32_t, usparse::usparse_fpc32, usparse::usparse_fpc64,
                               usparse::usparse_fpc64)
LINE_ENHANCE_INSTANTIATE_MIXED(usparse::usparse_fpc64, int64_t, int64_t, usparse::usparse_fpc32, usparse::usparse_fpc64,
                               usparse::usparse_fpc64)

#undef LINE_ENHANCE_INSTANTIATE_MIXED
#undef LINE_ENHANCE_INSTANTIATE
