//
// Created by chu genshen on 2026/1/13.
//

#include "csrmv_tuner_input.h"
#include <cstdlib>
#include <ctime>
#include <hip/hip_runtime.h>
#include <iostream>
#include <string>
#include <type_traits>

#include "common/usparse_utils.hpp"
#include "matrix_format/csr_binary_reader.hpp"
#include "matrix_format/csr_mtx_reader.hpp"
#include "matrix_format/matrix_market_reader.hpp"
#include "matrix_format/sparse_format.h"
#include "timer/timer.h"
#include "utils.hpp"
#include "utils/benchmark_main.hpp" // from dir: bechmarks
#include "verification.h"

#include "usparse_tuner.h"

namespace utuner {
  namespace cli {
    class CsrmvTunerCliPlugin;

    // the cli arguments
    utuner::cli::CsrmvTunerCliPlugin *cli_plugin_ptr = nullptr;

    class CsrmvTunerCliPlugin : public ubench::CliPlugin {
    private:
      args::ArgumentParser &parser;
      args::ValueFlag<std::string> cli_perf_summary_output;

    public:
      usparse::tuner::internal::PERF_RECORD_OUTPUT bench_summary_output_format =
          usparse::tuner::internal::PERF_RECORD_OUTPUT_STDOUT;

      // define cli argument
      CsrmvTunerCliPlugin(args::ArgumentParser &_parser)
          : ubench::CliPlugin(_parser), parser(_parser),
            cli_perf_summary_output(parser, "output", "performance summary output format: `stdout` (default) or `csv`",
                                    {'o', "output"}) {
        cli_plugin_ptr = this;
      }

      void cli_args_define() override {}

      // parse and get the cli argument
      void cli_args_parse() override {
        if (cli_perf_summary_output) {
          const std::string output_mode_str = args::get(cli_perf_summary_output);
          if (output_mode_str == "csv") {
            bench_summary_output_format = usparse::tuner::internal::PERF_RECORD_OUTPUT_CSV;
          } else if (output_mode_str == "stdout") {
            bench_summary_output_format = usparse::tuner::internal::PERF_RECORD_OUTPUT_STDOUT;
          } else {
            std::cerr << "warning: unsupported summary output format: " << output_mode_str
                      << ". Using default (stdout)." << std::endl;
          }
        }
      };
    };
  } // namespace cli
} // namespace utuner

template <typename INDEX_TYPE, typename DATA_TYPE>
void test_spmv(std::string mtx_path, csr_mtx<INDEX_TYPE, DATA_TYPE> h_csr, host_vectors<DATA_TYPE> h_vectors);

template <typename PRECISION>
int run_csr_benchmark(std::string mtx_path, csr_mtx<type_sparse_default_index, PRECISION> h_csr) {
  host_vectors<PRECISION> h_vectors{};
  create_host_data(h_csr, h_vectors, true);
  test_spmv<type_sparse_default_index, PRECISION>(mtx_path, h_csr, h_vectors);
  destroy_host_data(h_vectors);
  return 0;
}

utuner::csrmv_tuner_input<int, double> *global_csrmv_tuner_input_ptr = nullptr;

template <typename INDEX_TYPE, typename DATA_TYPE>
void test_spmv(std::string mtx_path, csr_mtx<INDEX_TYPE, DATA_TYPE> h_csr, host_vectors<DATA_TYPE> h_vectors) {
  USPARSE_HIP_API_CHECK(hipSetDevice(0));

  if constexpr (std::is_same<DATA_TYPE, float>::value) {
    std::cerr << "the tuner currently only supports FP64 (double). Exit." << std::endl;
    return;
  }

  DATA_TYPE *dev_x, *dev_y;
  csr_mtx<INDEX_TYPE, DATA_TYPE> d_csr = create_device_data(h_csr, h_vectors.hX, h_vectors.temphY, dev_x, dev_y);
  auto dev_csr_const = d_csr.as_const();

  USPARSE_HIP_API_CHECK(hipMemcpy(dev_y, h_vectors.temphY, h_csr.rows * sizeof(DATA_TYPE), hipMemcpyHostToDevice));

  // set global variable for passing parameter to tuner.
  utuner::csrmv_tuner_input<INDEX_TYPE, DATA_TYPE> _csrmv_tuner_input =
      utuner::csrmv_tuner_input<INDEX_TYPE, DATA_TYPE>{
          .mtx_path = mtx_path,
          .csr_mat = usparse::csr_desc<INDEX_TYPE, DATA_TYPE>(dev_csr_const),
          .x = dev_x,
          .y = dev_y,
      };

  if constexpr (std::is_same<DATA_TYPE, usparse::usparse_fp64>::value) {
    global_csrmv_tuner_input_ptr = &_csrmv_tuner_input;
  }

  auto out_format = usparse::tuner::internal::PERF_RECORD_OUTPUT_STDOUT;
  if (utuner::cli::cli_plugin_ptr != nullptr) {
    out_format = utuner::cli::cli_plugin_ptr->bench_summary_output_format;
  }

  usparse::tuner::internal::run_all_tuners();
  usparse::tuner::internal::print_header(out_format);
  usparse::tuner::internal::print_summary(out_format);
}

int main(int argc, char **argv) {
  return ubench::benchmark_main<ubench::BENCH_FORMAT_CSR, utuner::cli::CsrmvTunerCliPlugin>(argc, argv);
}
