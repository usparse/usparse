//
// Created by genshen on 2025/10/18.
//

#ifndef USPARSE_BENCHMARK_MAIN_HPP
#define USPARSE_BENCHMARK_MAIN_HPP

#include <clipp.h>

#include <iostream>
#include <string>

#include "matrix_format/csr_binary_reader.hpp"
#include "matrix_format/matrix_market_reader.hpp"
#include "matrix_format/sparse_format.h"

template <typename PRECISION>
int run_csr_benchmark(std::string mtx_path, csr_mtx<type_sparse_default_index, PRECISION> h_csr);

template <typename PRECISION>
int run_coo_benchmark(std::string mtx_path, coo_mtx<type_sparse_default_index, PRECISION> h_coo);

namespace ubench {
  constexpr int BENCH_FORMAT_CSR = 1;
  constexpr int BENCH_FORMAT_COO = 2;

  template <int BENCH_FORMAT, typename PRECISION> void read_mtx_and_run(std::string mtx_path, const std::string &fmt);

  // TODO: https://github.com/Taywee/args
  /**
   * @tparam BENCH_FORMAT the sparse matrix format used for benchmarking. e.g. BENCH_FORMAT for coomv is COO.
   */
  template <int BENCH_FORMAT> int benchmark_main(int argc, char **argv) {
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
      read_mtx_and_run<BENCH_FORMAT, float>(mtx_path, fmt);
    } else if (fp == "fp64") {
      std::cout << "benchmarking FP64 on " << mtx_path << std::endl;
      read_mtx_and_run<BENCH_FORMAT, double>(mtx_path, fmt);
    } else {
      std::cerr << "unsupported precision option." << std::endl;
      return -1;
    }
    return 0;
  }

  /**
   * @tparam BENCH_FORMAT the sparse matrix format used for benchmarking. e.g. BENCH_FORMAT for coomv is COO.
   */
  template <int BENCH_FORMAT, typename PRECISION> void read_mtx_and_run(std::string mtx_path, const std::string &fmt) {
    if (fmt == "bin2") {
      // value in bin2 must be double
      csr_binary_reader<type_sparse_default_index, PRECISION, double> csr_bin_reader;
      csr_bin_reader.load_mat(mtx_path);
      csr_bin_reader.close_stream();

      csr_mtx<type_sparse_default_index, PRECISION> h_csr;
      h_csr.rows = csr_bin_reader.rows();
      h_csr.cols = csr_bin_reader.cols();
      h_csr.nnz = csr_bin_reader.nnz();
      // just reuse memory in file reading step.
      csr_bin_reader.as_raw_ptr(h_csr.values, h_csr.col_index, h_csr.row_ptr);

      if constexpr (BENCH_FORMAT == BENCH_FORMAT_CSR) {
        run_csr_benchmark<PRECISION>(mtx_path, h_csr);
      } else {
        std::cerr << "unsupported matrix format .bin2." << std::endl;
        return;
      }
    } else if (fmt == "mtx") {
      matrix_market_reader<type_sparse_default_index, PRECISION> mm_reader;
      matrix_market<type_sparse_default_index, PRECISION> mm = mm_reader.load_mat(mtx_path);

      if constexpr (BENCH_FORMAT == BENCH_FORMAT_CSR) {
        csr_mtx<type_sparse_default_index, PRECISION> h_csr;
        h_csr = mm.to_csr();
        run_csr_benchmark<PRECISION>(mtx_path, h_csr);
      } else if (BENCH_FORMAT == BENCH_FORMAT_COO) {
        coo_mtx<type_sparse_default_index, PRECISION> h_coo = mm.to_coo();
        run_coo_benchmark<PRECISION>(mtx_path, h_coo);
      }
    } else {
      std::cerr << "unsupported matrix format." << std::endl;
    }
    return;
  }
} // namespace ubench

// BENCH_FORMAT can be ubench::BENCH_FORMAT_COO or ubench::BENCH_FORMAT_CSR
#define UBENCH_MAIN(BENCH_FORMAT)                                                                                      \
  int main(int argc, char **argv) { return ubench::benchmark_main<BENCH_FORMAT>(argc, argv); }

#endif // USPARSE_BENCHMARK_MAIN_HPP
