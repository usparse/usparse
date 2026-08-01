//
// Created by genshen on 2021/7/15.
//

#include "vector_row.h"
#include "building_config.h"
#include "common/macros.h"

#include "opt_double_buffer.hpp"
#include "vector_row.inl"
#include "vector_row_adaptive.hpp"
#include "vector_row_native.hpp"

template <typename DATA_TYPE>
void vec_row_sparse_spmv(int trans, const DATA_TYPE alpha, const DATA_TYPE beta,
                         const csr_desc<int, DATA_TYPE> d_csr_desc, const DATA_TYPE *x, DATA_TYPE *y) {
  //  const int avg_eles_per_row = ceil(d_csr_desc.nnz + 0.0 / m);
  VAR_FROM_CSR_DESC(DATA_TYPE, d_csr_desc);
  const int avg_eles_per_row = d_csr_desc.nnz / m;

#define NATIVE_VECTOR_KERNEL_WRAPPER(N, FP)                                                                            \
  (native_vector_row_kernel<N, 256, __WF_SIZE__, FP>)<<<512, 256>>>(m, alpha, beta, rowptr, colindex, value, x, y);

  if (avg_eles_per_row <= 4 || __WF_SIZE__ <= 2) {
    NATIVE_VECTOR_KERNEL_WRAPPER(2, DATA_TYPE);
  } else if (avg_eles_per_row <= 8 || __WF_SIZE__ <= 4) {
    NATIVE_VECTOR_KERNEL_WRAPPER(4, DATA_TYPE);
  } else if (avg_eles_per_row <= 16 || __WF_SIZE__ <= 8) {
    NATIVE_VECTOR_KERNEL_WRAPPER(8, DATA_TYPE);
  } else if (avg_eles_per_row <= 32 || __WF_SIZE__ <= 16) {
    NATIVE_VECTOR_KERNEL_WRAPPER(16, DATA_TYPE);
  } else if (avg_eles_per_row <= 64 || __WF_SIZE__ <= 32) {
    NATIVE_VECTOR_KERNEL_WRAPPER(32, DATA_TYPE);
  } else {
    NATIVE_VECTOR_KERNEL_WRAPPER(64, DATA_TYPE);
  }

#undef NATIVE_VECTOR_KERNEL_WRAPPER
}

template <typename DATA_TYPE>
void adaptive_vec_row_sparse_spmv(const int weight_block_0, const int weight_block_1, int trans, const DATA_TYPE alpha,
                                  const DATA_TYPE beta, const csr_desc<int, DATA_TYPE> d_csr_desc, const DATA_TYPE *x,
                                  DATA_TYPE *y) {
  VAR_FROM_CSR_DESC(DATA_TYPE, d_csr_desc);

  const int avg_eles_per_row = d_csr_desc.nnz / m;
  const int bp = static_cast<int>(round((16.0 * weight_block_0) / (weight_block_0 + weight_block_1)));
  const int rescaled_bp = std::min(16 - 1, std::max(1, bp)); // up and low boundary.
  ADAPTIVE_VECTOR_KERNEL_WRAPPER(2, rescaled_bp, DATA_TYPE); // N is data blocks.
}

template void adaptive_vec_row_sparse_spmv<double>(const int weight_block_0, const int weight_block_1, int trans,
                                                   const double alpha, const double beta,
                                                   const csr_desc<int, double> d_csr_desc, const double *x, double *y);

template void adaptive_vec_row_sparse_spmv<float>(const int weight_block_0, const int weight_block_1, int trans,
                                                   const float alpha, const float beta,
                                                   const csr_desc<int, float> d_csr_desc, const float *x, float *y);

template void vec_row_sparse_spmv<double>(int trans, const double alpha, const double beta,
                                          const csr_desc<int, double> d_csr_desc, const double *x, double *y);

template void vec_row_sparse_spmv<float>(int trans, const float alpha, const float beta,
                                         const csr_desc<int, float> d_csr_desc, const float *x, float *y);
