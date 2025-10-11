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

void test_csr_spmm(std::string &mtx_path, type_csr h_csr, const int M, const int K, const int N, dtype *host_B,
                   dtype *host_C);

int main(int argc, char **argv) {
  std::string mtx_path = "", fmt = "mtx";

  auto cli = (clipp::value("input file", mtx_path),
              clipp::option("-f", "--format")
                      .doc("input matrix format, can be  `mtx` (matrix market, default) or "
                           "`bin2` (csr binary v2)") &
                  clipp::value("format", fmt));

  if (!parse(argc, argv, cli)) {
    std::cout << clipp::make_man_page(cli, argv[0]);
    return 0;
  }

  type_csr h_csr;
  if (fmt == "bin2") {
    csr_binary_reader<int32_t, dtype> csr_bin_reader;
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
    std::vector<dtype> B_host = std::vector<dtype>(K * N);
    std::vector<dtype> C_host = std::vector<dtype>(M * N);

    test_csr_spmm(mtx_path, h_csr, M, K, N, B_host.data(), C_host.data());
  } else if (fmt == "mtx") {
    matrix_market_reader<int, dtype> mm_reader;
    matrix_market<int, dtype> mm = mm_reader.load_mat(mtx_path);
    h_csr = mm.to_csr();

    // A: M*K, B: K*N, C: M*N
    const int B_cols = h_csr.cols;
    const int M = h_csr.rows;
    const int K = h_csr.cols;
    const int N = B_cols;
    std::vector<dtype> B_host = std::vector<dtype>(K * N);
    std::vector<dtype> C_host = std::vector<dtype>(M * N);

    test_csr_spmm(mtx_path, h_csr, M, K, N, B_host.data(), C_host.data());
  } else {
    std::cerr << "unsupported matrix format." << std::endl;
  }

  return 0;
}

void test_csr_spmm(std::string &mtx_path, type_csr h_csr, const int M, const int K, const int N, dtype *host_B,
                   dtype *host_C) {
  usparse::var_csr_desc<int, dtype> dev_csr =
      usparse::var_csr_desc<int, dtype>(h_csr.rows, h_csr.cols, h_csr.nnz); // pointers are nullptr
  // we use malloc to set pointers of `dev_csr`
  ubench::ubench_malloc_csr_device<int, dtype>(&dev_csr, h_csr.rows, h_csr.cols, h_csr.nnz);
  ubench::ubench_copy_csr_H2D<int, dtype>(h_csr, &dev_csr);

  dtype alpha = 1.0;
  dtype beta = 0.0;

  dtype *dev_B = nullptr;
  dtype *dev_C = nullptr;
  ubench::ubench_malloc_and_init_host_dev_vector<int, dtype, false>(&host_B, &dev_B, K * N);
  HIP_CHECK(hipMalloc((void **)&dev_C, M * N * sizeof(dtype)))

  std::vector<dtype> C_host_check = std::vector<dtype>(M * N); // for storing results calculated by the host side

  statistics::print_csrmm_statistics_header();
  UsparseCsrMMVector vec;
  vec.test(mtx_path, "row-split", usparse::usparse_operation::operation_none, alpha, beta, h_csr, dev_csr, M, K, N,
           dev_B, dev_C, host_B, host_C, C_host_check.data());
  HIP_CHECK(hipDeviceSynchronize());

  RocSparseCsrMM roc;
  roc.test(mtx_path, "rocSparse", usparse::usparse_operation::operation_none, alpha, beta, h_csr, dev_csr, M, K, N,
           dev_B, dev_C, host_B, host_C, C_host_check.data());
  HIP_CHECK(hipDeviceSynchronize());

  ubench::ubench_free_device_csr<int, dtype>(dev_csr);
}
