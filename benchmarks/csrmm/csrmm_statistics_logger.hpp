//
// Created by genshen on 2025/10/09.
//

#ifndef USPARSE_BENCHMARK_CSRMM_STATISTICS_LOGGER_HPP
#define USPARSE_BENCHMARK_CSRMM_STATISTICS_LOGGER_HPP

#include <iostream>
#include <string>

#include "../utils/benchmark_time.h"
#include "verification.h"

namespace statistics {

  inline void print_csrmm_statistics_header() {
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

  template <typename I, typename T>
  void print_csrmm_statistics(const std::string &mtx_path, const std::string &strategy_name, const I rows, const I cols,
                              const I nnz, BenchmarkTime bmt, const VerifyResult<I, T> verify_result) {
    const double calc_time_bandwidth = 0.0;
    const double calc_time_gflops = 0.0;
    const double total_time_bandwidth = 0.0;
    const double total_time_gflops = 0.0;

    std::cout << "PERFORMANCE," << mtx_path << "," << strategy_name << "," << rows << "," << cols << "," << nnz << ","
              << (nnz + 0.0) / rows << "," << calc_time_bandwidth << "," << calc_time_gflops << ","
              << total_time_bandwidth << "," << total_time_gflops << "," << bmt.pre_time_use << "," << bmt.calc_time_use
              << "," << bmt.calc2_time_use << "," << bmt.destroy_time_use << "," << bmt.total_time_use << ","
              << verify_result.first_failed_at << "," << verify_result.failed_count << "," << verify_result.max_error
              << std::endl;
  }
} // namespace statistics

#endif // USPARSE_BENCHMARK_CSRMM_STATISTICS_LOGGER_HPP
