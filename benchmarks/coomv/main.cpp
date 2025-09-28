//
// Created by genshen on 2025/9/28.
//

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "clipp.h"
#include "usparse.h"

#include "coomv_statistics_logger.hpp"
#include "matrix_format/csr_binary_reader.hpp"
#include "matrix_format/csr_mtx_reader.hpp"
#include "matrix_format/matrix_market_reader.hpp"
#include "matrix_format/sparse_format.h"

#include "coo_spmv.hpp"

void test_spmv(std::string mtx_path, type_coo h_coo, host_vectors<dtype> h_vectors);

int main(int argc, char **argv) {
  std::string mtx_path = "", fmt = "mtx";

  auto cli = (clipp::value("input file", mtx_path),
              clipp::option("-f", "--format")
                      .doc("input matrix format, can be  `mtx` (matrix market, default) or "
                           "`bin2` (csr binary v2; TODO:)") &
                  clipp::value("format", fmt));

  if (!parse(argc, argv, cli)) {
    std::cout << clipp::make_man_page(cli, argv[0]);
    return 0;
  }

  type_coo h_coo;
  host_vectors<dtype> h_vectors{};

  if (fmt == "mtx") {
    matrix_market_reader<int, dtype> mm_reader;
    matrix_market<int, dtype> mm = mm_reader.load_mat(mtx_path);
    h_coo = mm.to_coo();

    create_host_data(h_coo, h_vectors, true);
    test_spmv(mtx_path, h_coo, h_vectors);
    destroy_host_data(h_vectors);
  } else {
    std::cerr << "unsupported matrix format." << std::endl;
  }
}

#define COOMV_BENCHMARK(instance, name, flag)                                                                          \
  {                                                                                                                    \
    instance ins;                                                                                                      \
    if (flag) {                                                                                                        \
      ins.test(mtx_path, name, operation, alpha, beta, h_coo_desc, d_coo_desc, h_vectors, dev_x, dev_y);               \
    }                                                                                                                  \
  }

void test_spmv(std::string mtx_path, type_coo h_coo, host_vectors<dtype> h_vectors) {
  HIP_CHECK(hipSetDevice(0));
  dtype *dev_x, *dev_y;
  type_coo d_coo = create_device_data(h_coo, h_vectors.hX, h_vectors.temphY, dev_x, dev_y);
  usparse::coo_desc<int, double> h_coo_desc =
      usparse::coo_desc<int, double>(h_coo.rows, h_coo.cols, h_coo.nnz, h_coo.row_index, h_coo.col_index, h_coo.values);
  usparse::coo_desc<int, double> d_coo_desc =
      usparse::coo_desc<int, double>(d_coo.rows, d_coo.cols, d_coo.nnz, d_coo.row_index, d_coo.col_index, d_coo.values);

  // set parameters
  enum usparse::usparse_operation operation = usparse::usparse_operation::operation_none;
  dtype alpha = 1.0;
  dtype beta = 1.0;

  statistics::print_coomv_statistics_header();
  COOMV_BENCHMARK(UsparseCooMVFlat, "usaprse-flat", true);

#ifdef __HIP_PLATFORM_HCC__
  // rocsparse
  COOMV_BENCHMARK(RocSparseCooMV, "rocSparse-coomv", true);
#endif
}
