//
// Created by genshen on 2025/10/09.
//

#include <iostream>
#include <string>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "usparse.h"

#include "../utils/benchmark_main.hpp"
#include "../utils/ubench_utils.h"
#include "utils.hpp"

#include "benchmark_csrmm.h"

UBENCH_MAIN(ubench::BENCH_FORMAT_CSR);

template <typename PRECISION>
void test_csr_spmm(std::string &mtx_path, csr_mtx<type_sparse_default_index, PRECISION> h_csr, const int M, const int K,
                   const int N, PRECISION *host_B, PRECISION *host_C);

template <typename PRECISION>
int run_csr_benchmark(std::string mtx_path, csr_mtx<type_sparse_default_index, PRECISION> h_csr) {
  // A: M*K, B: K*N, C: M*N
  const int B_cols = h_csr.cols;
  const int M = h_csr.rows;
  const int K = h_csr.cols;
  const int N = B_cols;
  std::vector<PRECISION> B_host = std::vector<PRECISION>(K * N);
  std::vector<PRECISION> C_host = std::vector<PRECISION>(M * N);

  test_csr_spmm(mtx_path, h_csr, M, K, N, B_host.data(), C_host.data());
  return 0;
}

template <typename PRECISION>
void test_csr_spmm(std::string &mtx_path, csr_mtx<type_sparse_default_index, PRECISION> h_csr, const int M, const int K,
                   const int N, PRECISION *host_B, PRECISION *host_C) {
  usparse::var_csr_desc<type_sparse_default_index, PRECISION> dev_csr =
      usparse::var_csr_desc<type_sparse_default_index, PRECISION>(h_csr.rows, h_csr.cols,
                                                                  h_csr.nnz); // pointers are nullptr
  // we use malloc to set pointers of `dev_csr`
  ubench::ubench_malloc_csr_device<type_sparse_default_index, PRECISION>(&dev_csr, h_csr.rows, h_csr.cols, h_csr.nnz);
  ubench::ubench_copy_csr_H2D<type_sparse_default_index, PRECISION>(h_csr, &dev_csr);

  PRECISION alpha = 1.0;
  PRECISION beta = 0.0;

  PRECISION *dev_B = nullptr;
  PRECISION *dev_C = nullptr;
  ubench::ubench_malloc_and_init_host_dev_vector<type_sparse_default_index, PRECISION, false>(&host_B, &dev_B, K * N);
  USPARSE_HIP_API_CHECK(hipMalloc((void **)&dev_C, M * N * sizeof(PRECISION)))

  std::vector<PRECISION> C_host_check =
      std::vector<PRECISION>(M * N); // for storing results calculated by the host side

  statistics::print_csrmm_statistics_header();
  UsparseCsrMMVector<type_sparse_default_index, PRECISION> vec;
  vec.test(mtx_path, "row-split", usparse::usparse_operation::operation_none, alpha, beta, h_csr, dev_csr, M, K, N,
           dev_B, dev_C, host_B, host_C, C_host_check.data());
  USPARSE_HIP_API_CHECK(hipDeviceSynchronize());

#ifdef __USPARSE_BUILD_BACKEND_ROCM__
  RocSparseCsrMM<type_sparse_default_index, PRECISION> roc;
  roc.test(mtx_path, "rocSparse", usparse::usparse_operation::operation_none, alpha, beta, h_csr, dev_csr, M, K, N,
           dev_B, dev_C, host_B, host_C, C_host_check.data());
  USPARSE_HIP_API_CHECK(hipDeviceSynchronize());
#endif

  ubench::ubench_free_device_csr<type_sparse_default_index, PRECISION>(dev_csr);
}
