//
// Created by genshen on 2025/9/28.
//

#ifndef USPARSE_BENCHMARK_COO_SPMV_STATISTICS_LOGGER_HPP
#define USPARSE_BENCHMARK_COO_SPMV_STATISTICS_LOGGER_HPP

#include <iostream>
#include <string>

#include "../utils/benchmark_time.h"
#include "verification.h"

namespace statistics {

  inline void print_coomv_statistics_header() {
    std::cout << "PERFORMANCE,"
              << "matrix name,"
              << "strategy name,"
              << "rows,"
              << "cols,"
              << "nnz,"
              << "nnz/row,"
              << "GB/s(calc_time),"
              << "GFLOPS(calc_time),"
              << "GB/s(total_time),"
              << "GFLOPS(total_time),"
              << "mid pre cost,"
              << "mid calc cost,"
              << "mid fixup(calc2) cost,"
              << "mid destroy cost,"
              << "mid total cost,"
              << "first_failed_at,"
              << "failed_count,"
              << "max_error" << std::endl;
  }

  template <typename T>
  void print_coomv_statistics(const std::string &mtx_path, const std::string &strategy_name, const int rows,
                              const int cols, const int nnz, BenchmarkTime bmt,
                              const VerifyResult<int, T> verify_result) {
    double mem_bytes = static_cast<double>(sizeof(T) * (2 * rows + nnz) + sizeof(int) * (nnz + nnz));

    double calc_time_bandwidth = (mem_bytes + 0.0) / (1024 * 1024 * 1024) / (bmt.calc_time_use / 1e3 / 1e3);
    double calc_time_gflops = static_cast<double>(2 * nnz) / bmt.calc_time_use / 1e3;

    double total_time_bandwidth = (mem_bytes + 0.0) / (1024 * 1024 * 1024) / (bmt.total_time_use / 1e3 / 1e3);
    double total_time_gflops = static_cast<double>(2 * nnz) / bmt.total_time_use / 1e3;

    std::cout << "PERFORMANCE," << mtx_path << "," << strategy_name << "," << rows << "," << cols << "," << nnz << ","
              << (nnz + 0.0) / rows << "," << calc_time_bandwidth << "," << calc_time_gflops << ","
              << total_time_bandwidth << "," << total_time_gflops << "," << bmt.pre_time_use << "," << bmt.calc_time_use
              << "," << bmt.calc2_time_use << "," << bmt.destroy_time_use << "," << bmt.total_time_use << ","
              << verify_result.first_failed_at << "," << verify_result.failed_count << "," << verify_result.max_error
              << std::endl;
  }
} // namespace statistics

#endif // USPARSE_BENCHMARK_COO_SPMV_STATISTICS_LOGGER_HPP
