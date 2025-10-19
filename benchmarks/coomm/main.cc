//
// Created by genshen on 2025/10/19.
//

#include <iostream>
#include <string>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "usparse.h"

#include "../utils/benchmark_main.hpp"
#include "../utils/ubench_utils.h"
#include "utils.hpp"

#include "benchmark_coomm.h"

UBENCH_MAIN(ubench::BENCH_FORMAT_COO);

template <typename PRECISION>
void test_coo_spmm(std::string &mtx_path, usparse::coo_desc<type_sparse_default_index, PRECISION> h_coo, const int M,
                   const int K, const int N, PRECISION *host_B, PRECISION *host_C);

template <typename PRECISION>
int run_coo_benchmark(std::string mtx_path, coo_mtx<type_sparse_default_index, PRECISION> h_coo) {
  // A: M*K, B: K*N, C: M*N
  const int B_cols = h_coo.cols;
  const int M = h_coo.rows;
  const int K = h_coo.cols;
  const int N = B_cols;
  std::vector<PRECISION> B_host = std::vector<PRECISION>(K * N);
  std::vector<PRECISION> C_host = std::vector<PRECISION>(M * N);

  usparse::coo_desc<type_sparse_default_index, PRECISION> usparse_host_coo =
      usparse::coo_desc<type_sparse_default_index, PRECISION>(h_coo.rows, h_coo.cols, h_coo.nnz, h_coo.row_index,
                                                              h_coo.col_index, h_coo.values);
  test_coo_spmm(mtx_path, usparse_host_coo, M, K, N, B_host.data(), C_host.data());
  return 0;
}

template <typename PRECISION>
void test_coo_spmm(std::string &mtx_path, usparse::coo_desc<type_sparse_default_index, PRECISION> h_coo, const int M,
                   const int K, const int N, PRECISION *host_B, PRECISION *host_C) {
  usparse::var_coo_desc<type_sparse_default_index, PRECISION> dev_coo =
      usparse::var_coo_desc<type_sparse_default_index, PRECISION>(h_coo.rows, h_coo.cols,
                                                                  h_coo.nnz); // pointers are nullptr
  // we use malloc to set pointers of `dev_coo`
  ubench::ubench_malloc_coo_device<type_sparse_default_index, PRECISION>(&dev_coo, h_coo.rows, h_coo.cols, h_coo.nnz);
  ubench::ubench_copy_coo_H2D<type_sparse_default_index, PRECISION>(h_coo, &dev_coo);

  PRECISION alpha = 1.0;
  PRECISION beta = 0.0;

  PRECISION *dev_B = nullptr;
  PRECISION *dev_C = nullptr;
  ubench::ubench_malloc_and_init_host_dev_vector<type_sparse_default_index, PRECISION, false>(&host_B, &dev_B, K * N);
  HIP_CHECK(hipMalloc((void **)&dev_C, M * N * sizeof(PRECISION)))

  std::vector<PRECISION> C_host_check =
      std::vector<PRECISION>(M * N); // for storing results calculated by the host side

  statistics::print_coomm_statistics_header();
  usparse::coo_desc<type_sparse_default_index, PRECISION> usparse_dev_coo =
      usparse::coo_desc<type_sparse_default_index, PRECISION>(dev_coo.rows, dev_coo.cols, dev_coo.nnz,
                                                              dev_coo.row_index, dev_coo.col_index, dev_coo.values);
  // todo: keep the same style as coomm

  UsparseCooMMVector<type_sparse_default_index, PRECISION> vec;
  vec.test(mtx_path, "row-split", usparse::usparse_operation::operation_none, alpha, beta, h_coo,
           usparse_dev_coo, M, K, N, dev_B, dev_C, host_B, host_C, C_host_check.data());
  HIP_CHECK(hipDeviceSynchronize());

  ubench::ubench_free_device_coo<type_sparse_default_index, PRECISION>(dev_coo);
}
