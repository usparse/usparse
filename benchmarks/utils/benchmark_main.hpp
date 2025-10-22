//
// Created by genshen on 2025/10/18.
//

#ifndef USPARSE_BENCHMARK_MAIN_HPP
#define USPARSE_BENCHMARK_MAIN_HPP

#include <iostream>
#include <string>

#include <args.hxx>

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
    args::ArgumentParser parser("This is usparse benchmark.", "authors: genshen.");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<std::string> mtx_path(parser, "matrix", "The matrix file", {'m', "mtx"});
    args::ValueFlag<std::string> fmt(
        parser, "format", "The input matrix format. can be  `mtx` (matrix market, default) or `bin2` (csr binary v2)",
        {'f', "fmt"});
    args::ValueFlag<std::string> fp(parser, "precision",
                                    "The float percision used in benchmark. can be `fp32` or `fp64` (default)", {"fp"});

    try {
      parser.ParseCLI(argc, (const char *const *)argv);
    } catch (args::Help) {
      std::cout << parser;
      return false;
    } catch (args::ParseError e) {
      std::cerr << e.what() << std::endl;
      std::cerr << parser;
      return false;
    } catch (args::ValidationError e) {
      std::cerr << e.what() << std::endl;
      std::cerr << parser;
      return false;
    }

    std::string mtx_path_str = "", fmt_str = "mtx", fp_str = "fp64";
    if (mtx_path) {
      mtx_path_str = args::get(mtx_path);
    } else {
      std::cerr << "matrix path must be specified." << std::endl;
      std::cerr << parser;
      return 1;
    }
    if (fmt) {
      fmt_str = args::get(fmt);
    }
    if (fp) {
      fp_str = args::get(fp);
    }

    if (fp_str == "fp32") {
      std::cout << "benchmarking FP32 on " << mtx_path_str << std::endl;
      read_mtx_and_run<BENCH_FORMAT, float>(mtx_path_str, fmt_str);
    } else if (fp_str == "fp64") {
      std::cout << "benchmarking FP64 on " << mtx_path_str << std::endl;
      read_mtx_and_run<BENCH_FORMAT, double>(mtx_path_str, fmt_str);
    } else {
      std::cerr << "unsupported precision option: " << fp_str << "." << std::endl;
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
        coo_mtx<type_sparse_default_index, PRECISION> h_coo;
        // malloc new memory for coo matrix. we dont reuse it.
        h_coo.alloc(h_csr.rows, h_csr.cols, h_csr.nnz);
        h_csr.to_coo(h_coo);
        run_coo_benchmark<PRECISION>(mtx_path, h_coo);
        h_coo.destroy();
      }
      // csr_bin_reader.unload_mat(); // release memory
      h_csr.destroy();
    } else if (fmt == "mtx") {
      matrix_market_reader<type_sparse_default_index, PRECISION> mm_reader;
      // todo: use alloc to replace `new` array for mm.
      matrix_market<type_sparse_default_index, PRECISION> mm = mm_reader.load_mat(mtx_path);

      if constexpr (BENCH_FORMAT == BENCH_FORMAT_CSR) {
        csr_mtx<type_sparse_default_index, PRECISION> h_csr;
        h_csr = mm.to_csr(); // TODO: avoid alloc inside to_csr
        run_csr_benchmark<PRECISION>(mtx_path, h_csr);
        h_csr.destroy();
      } else if (BENCH_FORMAT == BENCH_FORMAT_COO) {
        // TODO: avoid alloc inside to_coo
        coo_mtx<type_sparse_default_index, PRECISION> h_coo = mm.to_coo();
        run_coo_benchmark<PRECISION>(mtx_path, h_coo);
        h_coo.destroy();
      }
      mm.destroy();
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
