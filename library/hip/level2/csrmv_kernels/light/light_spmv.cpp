//
// Created by genshen on 2021/7/15.
//

#include "spmv_hip_acc_imp.h"

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "usparse_types.h"

#include "building_config.h"
#include "common/macros.h"
#include "common/usparse_utils.hpp"

#define LIGHT_KERNEL_CALLER(N)                                                                                         \
  ((spmv_light_kernel<N, __WF_SIZE__, DATA_TYPE>)<<<256, 256>>> (m, alpha, beta, hip_row_counter, rowptr, colindex, value, x, y))

template <typename INDEX_TYPE, typename DATA_TYPE>
void light_sparse_spmv(int trans, const DATA_TYPE alpha, const DATA_TYPE beta,
                       const csr_desc<INDEX_TYPE, DATA_TYPE> d_csr_desc, const DATA_TYPE *x, DATA_TYPE *y) {
  VAR_FROM_CSR_DESC(DATA_TYPE, d_csr_desc);

  int *hip_row_counter;
  USPARSE_HIP_API_CHECK(hipMalloc((void **)&hip_row_counter, sizeof(int)));
  const int avg_eles_per_row = d_csr_desc.nnz / m;
  USPARSE_HIP_API_CHECK(hipMemset(hip_row_counter, 0, sizeof(int)));

  if (avg_eles_per_row <= 2) {
    LIGHT_KERNEL_CALLER(1);
  } else if (avg_eles_per_row <= 4 || __WF_SIZE__ <= 2) {
    LIGHT_KERNEL_CALLER(2);
  } else if (avg_eles_per_row <= 8 || __WF_SIZE__ <= 4) {
    LIGHT_KERNEL_CALLER(4);
  } else if (avg_eles_per_row <= 16 || __WF_SIZE__ <= 8) {
    LIGHT_KERNEL_CALLER(8);
  } else if (avg_eles_per_row <= 32 || __WF_SIZE__ <= 16) {
    LIGHT_KERNEL_CALLER(16);
  } else if (avg_eles_per_row <= 64 || __WF_SIZE__ <= 32) {
    LIGHT_KERNEL_CALLER(32);
  } else {
    LIGHT_KERNEL_CALLER(64);
  }

  USPARSE_HIP_API_CHECK(hipFree(hip_row_counter));
}

template void light_sparse_spmv<usparse::usparse_int, usparse::usparse_fp32>(
    int trans, const usparse::usparse_fp32 alpha, const usparse::usparse_fp32 beta,
    const csr_desc<usparse::usparse_int, usparse::usparse_fp32> d_csr_desc, const usparse::usparse_fp32 *x,
    usparse::usparse_fp32 *y);

template void light_sparse_spmv<usparse::usparse_int, usparse::usparse_fp64>(
    int trans, const usparse::usparse_fp64 alpha, const usparse::usparse_fp64 beta,
    const csr_desc<usparse::usparse_int, usparse::usparse_fp64> d_csr_desc, const usparse::usparse_fp64 *x,
    usparse::usparse_fp64 *y);
