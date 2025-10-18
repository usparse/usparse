//
// Created by genshen on 2025/10/09.
//

#include <iostream>
#include <string>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "usparse.h"

#include "../utils/ubench_utils.h"
#include "clipp.h"
#include "matrix_format/csr_binary_reader.hpp"
#include "matrix_format/csr_mtx_reader.hpp"
#include "matrix_format/matrix_market_reader.hpp"
#include "matrix_format/sparse_format.h"
#include "utils.hpp"

#include "benchmark_csrmm.h"

template <typename PRECISION>
void test_csr_spmm(std::string &mtx_path, csr_mtx<type_sparse_default_index, PRECISION> h_csr, const int M, const int K,
                   const int N, PRECISION *host_B, PRECISION *host_C);

template <typename PRECISION> void read_mtx_and_run(std::string mtx_path, const std::string &fmt);

int main(int argc, char **argv) {
  std::string mtx_path = "", fmt = "mtx";
  std::string fp = "fp64";

  auto cli = (clipp::value("input file", mtx_path),
              clipp::option("-f", "--format")
                      .doc("input matrix format, can be  `mtx` (matrix market, default) or "
                           "`bin2` (csr binary v2)") &
                  clipp::value("format", fmt),
              clipp::option("-p", "--precision").doc("the percision used in benchmark, can be `fp32` or `fp64`") &
                  clipp::value("precision", fp));

  if (!parse(argc, argv, cli)) {
    std::cout << clipp::make_man_page(cli, argv[0]);
    return 0;
  }

  if (fp == "fp32") {
    std::cout << "benchmarking FP32 on " << mtx_path << std::endl;
    read_mtx_and_run<float>(mtx_path, fmt);
  } else if (fp == "fp64") {
    std::cout << "benchmarking FP64 on " << mtx_path << std::endl;
    read_mtx_and_run<double>(mtx_path, fmt);
  } else {
    std::cerr << "unsupported precision option." << std::endl;
    return -1;
  }
  return 0;
}

template <typename PRECISION> void read_mtx_and_run(std::string mtx_path, const std::string &fmt) {
  csr_mtx<type_sparse_default_index, PRECISION> h_csr;
  if (fmt == "bin2") {
    // TODO: value in bin2 must be double.
    csr_binary_reader<type_sparse_default_index, PRECISION, double> csr_bin_reader;
    csr_bin_reader.load_mat(mtx_path);
    csr_bin_reader.close_stream();

    h_csr.rows = csr_bin_reader.rows();
    h_csr.cols = csr_bin_reader.cols();
    h_csr.nnz = csr_bin_reader.nnz();
    // just reuse memory in file reading step.
    csr_bin_reader.as_raw_ptr(h_csr.values, h_csr.col_index, h_csr.row_ptr);

    // A: M*K, B: K*N, C: M*N
    const int B_cols = h_csr.cols;
    const int M = h_csr.rows;
    const int K = h_csr.cols;
    const int N = B_cols;
    std::vector<PRECISION> B_host = std::vector<PRECISION>(K * N);
    std::vector<PRECISION> C_host = std::vector<PRECISION>(M * N);

    test_csr_spmm(mtx_path, h_csr, M, K, N, B_host.data(), C_host.data());
  } else if (fmt == "mtx") {
    matrix_market_reader<type_sparse_default_index, PRECISION> mm_reader;
    matrix_market<type_sparse_default_index, PRECISION> mm = mm_reader.load_mat(mtx_path);
    h_csr = mm.to_csr();

    // A: M*K, B: K*N, C: M*N
    const int B_cols = h_csr.cols;
    const int M = h_csr.rows;
    const int K = h_csr.cols;
    const int N = B_cols;
    std::vector<PRECISION> B_host = std::vector<PRECISION>(K * N);
    std::vector<PRECISION> C_host = std::vector<PRECISION>(M * N);

    test_csr_spmm(mtx_path, h_csr, M, K, N, B_host.data(), C_host.data());
  } else {
    std::cerr << "unsupported matrix format." << std::endl;
  }
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
  HIP_CHECK(hipMalloc((void **)&dev_C, M * N * sizeof(PRECISION)))

  std::vector<PRECISION> C_host_check =
      std::vector<PRECISION>(M * N); // for storing results calculated by the host side

  statistics::print_csrmm_statistics_header();
  UsparseCsrMMVector<type_sparse_default_index, PRECISION> vec;
  vec.test(mtx_path, "row-split", usparse::usparse_operation::operation_none, alpha, beta, h_csr, dev_csr, M, K, N,
           dev_B, dev_C, host_B, host_C, C_host_check.data());
  HIP_CHECK(hipDeviceSynchronize());

  RocSparseCsrMM<type_sparse_default_index, PRECISION> roc;
  roc.test(mtx_path, "rocSparse", usparse::usparse_operation::operation_none, alpha, beta, h_csr, dev_csr, M, K, N,
           dev_B, dev_C, host_B, host_C, C_host_check.data());
  HIP_CHECK(hipDeviceSynchronize());

  ubench::ubench_free_device_csr<type_sparse_default_index, PRECISION>(dev_csr);
}
