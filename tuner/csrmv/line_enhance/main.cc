//
// Created by chu genshen on 2026/1/13.
//

#include <cstdlib>
#include <ctime>
#include <string>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "csrmv_tuner_input.h"

#include "matrix_format/csr_binary_reader.hpp"
#include "matrix_format/csr_mtx_reader.hpp"
#include "matrix_format/matrix_market_reader.hpp"
#include "matrix_format/sparse_format.h"
#include "timer/timer.h"
#include "utils.hpp"
#include "utils/benchmark_main.hpp" // from dir: bechmarks
#include "verification.h"

#include "usparse_tuner.h"

UBENCH_MAIN(ubench::BENCH_FORMAT_CSR);

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

// todo: more elegent args passing.
utuner::csrmv_tuner_input<int, double> *global_csrmv_tuner_input_ptr;

template <typename INDEX_TYPE, typename DATA_TYPE>
void test_spmv(std::string mtx_path, csr_mtx<INDEX_TYPE, DATA_TYPE> h_csr, host_vectors<DATA_TYPE> h_vectors) {
  HIP_CHECK(hipSetDevice(0));

  if constexpr (std::is_same<DATA_TYPE, usparse::usparse_fp32>::value) {
    printf("only FP64 is supported.\n");
    return;
  }

  DATA_TYPE *dev_x, *dev_y;
  csr_mtx<INDEX_TYPE, DATA_TYPE> d_csr = create_device_data(h_csr, h_vectors.hX, h_vectors.temphY, dev_x, dev_y);
  auto dev_csr_const = d_csr.as_const();

  HIP_CHECK(hipMemcpy(dev_y, h_vectors.temphY, h_csr.rows * sizeof(DATA_TYPE), hipMemcpyHostToDevice));

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

  // run usparse tuner
  usparse::tuner::internal::run_all_tuners();
  usparse::tuner::internal::print_header();
  usparse::tuner::internal::print_summary();
  return;
}
