//
// Created by genshen on 2025/10/09.
//

#ifndef USPARSE_BENCHMARK_CSR_SPMM
#define USPARSE_BENCHMARK_CSR_SPMM

#include <cstdlib>
#include <ctime>
#include <string>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "include/usparse.h"
#include "include/usparse_matrix.h"

#include "csrmm_statistics_logger.hpp"
#include "utils.hpp"
#include "verification.h"
using dtype = double;

template <typename I, typename T>
void host_csr_spmm(const T alpha, const T beta, const I *csr_row_ptr, const I *csr_col_index, const T *csr_values,
                   const I A_rows, const I A_cols, const I A_nnz, const I M, const I K, const I N, const T *B_host,
                   T *C_host_check) {
  memset(C_host_check, 0, M * N * sizeof(T));

  for (I i = 0; i < M; i++) {
    I row_start = csr_row_ptr[i];
    I row_end = csr_row_ptr[i + 1];

    for (I j = row_start; j < row_end; j++) {
      I col_idx = csr_col_index[j];
      double a_val = csr_values[j];

      for (I k = 0; k < N; k++) {
        C_host_check[i * N + k] += a_val * B_host[col_idx * N + k];
        // if (i *N + k == 7517) {
        //   printf("host: C[%d,%d] += A[%d,%d] * B[%d, %d]: delta=%f=%f*%f\n",
        //     i, k, i, col_idx, col_idx, k, a_val * B_host[col_idx * N + k], a_val, B_host[col_idx * N + k]);
        // }
      }
    }
  }
  // printf("host_check: C[0]: %f\n", C_host_check[0]);
}

struct CsrSpMM {
  /**
   * function returning a flag for if to support alpha and beta.
   * true: calculate C = alpha*A*B + beta*C, false calculate: C = A*B
   */
  virtual bool verify_beta_c() = 0;

  /**
   *
   * @param alpha alpha in C = alpha*A*B + beta*C
   * @param beta beta in C = alpha*A*B + beta*C
   * @param h_csr_desc csr description on host.
   * @param d_csr_desc csr description on device.
   * @param B device matrix B.
   * @param C device matrix C.
   * @param bmt spmm elapsed time
   */
  virtual void csr_spmm_impl(int trans, const dtype alpha, const dtype beta, const csr_desc<int, dtype> h_csr_desc,
                             const csr_desc<int, dtype> d_csr_desc, const int M, const int K, const int N,
                             dtype *B_dev, dtype *C_dev, BenchmarkTime *bmt) = 0;

  void test(std::string mtx_path, const std::string &strategy_name, enum usparse::usparse_operation operation,
            dtype alpha, dtype beta, csr_desc<int, dtype> h_csr, csr_desc<int, dtype> d_csr, const int M, const int K,
            const int N, dtype *B_dev, dtype *C_dev, dtype *B_host, dtype *C_host, dtype *C_host_check) {
    BenchmarkTimeArray bmt_array;
    // warm up GPU
    for (int i = 0; i < 10; ++i) {
      // call sparse spmv
      csr_spmm_impl(operation, alpha, beta, h_csr, d_csr, M, K, N, B_dev, C_dev, nullptr);
    }
    hipDeviceSynchronize();

    // execute device SpMV
    for (int i = 0; i < BENCHMARK_ARRAY_SIZE; i++) {
      BenchmarkTime bmt;
      csr_spmm_impl(operation, alpha, beta, h_csr, d_csr, M, K, N, B_dev, C_dev, &bmt);
      hipDeviceSynchronize();
      hipMemset(C_dev, 0, M * N * sizeof(dtype));
      bmt_array.append(bmt);
    }

    // device result check
    csr_spmm_impl(operation, alpha, beta, h_csr, d_csr, M, K, N, B_dev, C_dev, nullptr);
    hipDeviceSynchronize();
    HIP_CHECK(hipMemcpy(C_host, C_dev, M * N * sizeof(dtype), hipMemcpyDeviceToHost));

    // host side verification
    if (verify_beta_c()) {
      // C = alpha*A*B + beta*C
      host_csr_spmm<int, dtype>(alpha, beta, h_csr.row_ptr, h_csr.col_index, h_csr.values, h_csr.rows, h_csr.cols,
                                h_csr.nnz, M, K, N, B_host, C_host_check);
    } else {
      // C = A*B
      host_csr_spmm<int, dtype>(1.0, 0.0, h_csr.row_ptr, h_csr.col_index, h_csr.values, h_csr.rows, h_csr.cols,
                                h_csr.nnz, M, K, N, B_host, C_host_check);
    }

    const VerifyResult<int, double> verify_result = verify_y(C_host, C_host_check, M * N);
    statistics::print_csrmm_statistics<dtype>(mtx_path, strategy_name, h_csr.rows, h_csr.cols, h_csr.nnz,
                                              bmt_array.get_mid_time(), verify_result);
  }
};

#endif // USPARSE_BENCHMARK_CSR_SPMM
