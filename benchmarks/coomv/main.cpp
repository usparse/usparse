//
// Created by genshen on 2025/9/28.
//

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "usparse.h"

#include "coomv_statistics_logger.hpp"
#include "matrix_format/csr_binary_reader.hpp"
#include "matrix_format/csr_mtx_reader.hpp"
#include "matrix_format/matrix_market_reader.hpp"
#include "matrix_format/sparse_format.h"

#include "../utils/benchmark_main.hpp"
#include "coo_spmv.hpp"
#include "utils.hpp"

UBENCH_MAIN(ubench::BENCH_FORMAT_COO);

template <typename INDEX_TYPE, typename DATA_TYPE>
void test_spmv(std::string mtx_path, coo_mtx<INDEX_TYPE, DATA_TYPE> h_coo, host_vectors<DATA_TYPE> h_vectors);

template <typename PRECISION>
int run_coo_benchmark(std::string mtx_path, coo_mtx<type_sparse_default_index, PRECISION> h_coo) {
  host_vectors<PRECISION> h_vectors{};

  create_host_data(h_coo, h_vectors, true);
  test_spmv<type_sparse_default_index, PRECISION>(mtx_path, h_coo, h_vectors);
  destroy_host_data(h_vectors);
  return 0;
}

#define COOMV_BENCHMARK(instance, name, flag)                                                                          \
  {                                                                                                                    \
    instance<INDEX_TYPE, DATA_TYPE> ins;                                                                               \
    if (flag) {                                                                                                        \
      ins.test(mtx_path, name, operation, alpha, beta, h_coo_desc, d_coo_desc, h_vectors, dev_x, dev_y);               \
    }                                                                                                                  \
  }

template <typename INDEX_TYPE, typename DATA_TYPE>
void test_spmv(std::string mtx_path, coo_mtx<INDEX_TYPE, DATA_TYPE> h_coo, host_vectors<DATA_TYPE> h_vectors) {
  USPARSE_HIP_API_CHECK(hipSetDevice(0));
  DATA_TYPE *dev_x, *dev_y;
  coo_mtx<INDEX_TYPE, DATA_TYPE> d_coo =
      create_device_data<INDEX_TYPE, DATA_TYPE>(h_coo, h_vectors.hX, h_vectors.temphY, dev_x, dev_y);
  usparse::coo_desc<INDEX_TYPE, DATA_TYPE> h_coo_desc = usparse::coo_desc<INDEX_TYPE, DATA_TYPE>(
      h_coo.rows, h_coo.cols, h_coo.nnz, h_coo.row_index, h_coo.col_index, h_coo.values);
  usparse::coo_desc<INDEX_TYPE, DATA_TYPE> d_coo_desc = usparse::coo_desc<INDEX_TYPE, DATA_TYPE>(
      d_coo.rows, d_coo.cols, d_coo.nnz, d_coo.row_index, d_coo.col_index, d_coo.values);

  // set parameters
  enum usparse::usparse_operation operation = usparse::usparse_operation::operation_none;
  DATA_TYPE alpha = 1.0;
  DATA_TYPE beta = 1.0;

  statistics::print_coomv_statistics_header();
  COOMV_BENCHMARK(UsparseCoomvNaiveFlat, "usparse-naive-flat", true);
  COOMV_BENCHMARK(UsparseCoomvVector, "usparse-vector", true);
  COOMV_BENCHMARK(UsparseCoomvFlat, "usparse-flat", true);
  COOMV_BENCHMARK(UsparseCoomvLine_Enhance, "usparse-line-enhance", true);

#ifdef __HIP_PLATFORM_HCC__
  // rocsparse
  COOMV_BENCHMARK(RocSparseCooMV, "rocSparse-coomv", true);
#endif // todo: destroy device data.
}
