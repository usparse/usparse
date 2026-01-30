//
// Created by genshen on 2025/9/28.
//

#ifndef USPARSE_BENCHMARK_COO_SPMV
#define USPARSE_BENCHMARK_COO_SPMV

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "../utils/benchmark_time.h"
#include "coomv_statistics_logger.hpp"
#include "usparse_matrix.h"
#include "utils.hpp"
#include "verification.h"

/**
 * The base class for Coomv benchmark.
 * each algorithm is based on this class and implement the coo_spmv_impl function.
 * @tparam INDEX_TYPE type for index.
 * @tparam DATA_TYPE type for data in the sparse matrix.
 */
template <typename INDEX_TYPE, typename DATA_TYPE> struct CooSpMV {
  /**
   * function returning a flag for if to support alpha and beta.
   * true: calculate y = alpha*A*x + beta*y, false calculate: y = A*x
   */
  virtual bool verify_beta_y() = 0;

  /**
   *
   * @param trans Matrix transpose. current only support operation_none.
   * @param alpha alpha in y = alpha*A*x + beta*y
   * @param beta beta in y = alpha*A*x + beta*y
   * @param h_rowptr row offset on host size.
   * @param d_csr_desc csr description.
   * @param x device vector x.
   * @param y device vector y.
   * @param bmt spmv elapsed time
   */
  virtual void coo_spmv_impl(int trans, const DATA_TYPE alpha, const DATA_TYPE beta,
                             const usparse::coo_desc<INDEX_TYPE, DATA_TYPE> h_coo_desc,
                             const usparse::coo_desc<INDEX_TYPE, DATA_TYPE> d_coo_desc, const DATA_TYPE *x,
                             DATA_TYPE *y, BenchmarkTime *bmt) = 0;

  void test(std::string mtx_path, const std::string &strategy_name, enum usparse::usparse_operation operation,
            DATA_TYPE alpha, DATA_TYPE beta, usparse::coo_desc<INDEX_TYPE, DATA_TYPE> h_coo_desc,
            usparse::coo_desc<INDEX_TYPE, DATA_TYPE> d_coo_desc, host_vectors<DATA_TYPE> h_vectors, DATA_TYPE *&dev_x,
            DATA_TYPE *&dev_y) {
    BenchmarkTimeArray bmt_array;
    // warm up GPU
    for (int i = 0; i < 10; ++i) {
      // call sparse spmv
      USPARSE_HIP_API_CHECK(
          hipMemcpy(dev_y, h_vectors.temphY, d_coo_desc.rows * sizeof(DATA_TYPE), hipMemcpyHostToDevice))
      try {
        coo_spmv_impl(operation, alpha, beta, h_coo_desc, d_coo_desc, dev_x, dev_y, nullptr);
      } catch (const std::runtime_error &error) {
        std::cout << "matrix name: " << mtx_path << ", strategy name: " << strategy_name << std::endl;
        std::cout << "error occur, return" << std::endl;
        std::cerr << error.what() << std::endl;
        USPARSE_HIP_API_CHECK(hipDeviceSynchronize());
        memcpy(h_vectors.hhY, h_vectors.temphY, d_coo_desc.rows * sizeof(DATA_TYPE));
        USPARSE_HIP_API_CHECK(
            hipMemcpy(dev_y, h_vectors.temphY, d_coo_desc.rows * sizeof(DATA_TYPE), hipMemcpyHostToDevice))
        return;
      }
    }
    USPARSE_HIP_API_CHECK(hipDeviceSynchronize());

    // execute device SpMV
    for (int i = 0; i < BENCHMARK_ARRAY_SIZE; i++) {
      BenchmarkTime bmt;
      USPARSE_HIP_API_CHECK(
          hipMemcpy(dev_y, h_vectors.temphY, d_coo_desc.rows * sizeof(DATA_TYPE), hipMemcpyHostToDevice))
      USPARSE_HIP_API_CHECK(hipDeviceSynchronize());
      coo_spmv_impl(operation, alpha, beta, h_coo_desc, d_coo_desc, dev_x, dev_y, &bmt);
      USPARSE_HIP_API_CHECK(hipDeviceSynchronize());
      bmt_array.append(bmt);
    }

    // device result check
    USPARSE_HIP_API_CHECK(
        hipMemcpy(dev_y, h_vectors.temphY, h_coo_desc.rows * sizeof(DATA_TYPE), hipMemcpyHostToDevice))
    coo_spmv_impl(operation, alpha, beta, h_coo_desc, d_coo_desc, dev_x, dev_y, nullptr);
    USPARSE_HIP_API_CHECK(hipDeviceSynchronize());
    USPARSE_HIP_API_CHECK(hipMemcpy(h_vectors.hY, dev_y, d_coo_desc.rows * sizeof(DATA_TYPE), hipMemcpyDeviceToHost));

    // host side verification
    if (verify_beta_y()) {
      // y = alpha*A*x + beta*y
      host_coo_spmv(alpha, beta, h_coo_desc.values, h_coo_desc.row_index, h_coo_desc.col_index, h_coo_desc.rows,
                    h_coo_desc.cols, h_coo_desc.nnz, h_vectors.hX, h_vectors.hhY);
    } else {
      // y = A*x
      host_coo_spmv(h_coo_desc.values, h_coo_desc.row_index, h_coo_desc.col_index, h_coo_desc.rows, h_coo_desc.cols,
                    h_coo_desc.nnz, h_vectors.hX, h_vectors.hhY);
    }

    const VerifyResult<INDEX_TYPE, DATA_TYPE> verify_result = verify_y(h_vectors.hY, h_vectors.hhY, h_coo_desc.rows);
    statistics::print_coomv_statistics<DATA_TYPE>(mtx_path, strategy_name, h_coo_desc.rows, h_coo_desc.cols,
                                                  h_coo_desc.nnz, bmt_array.get_mid_time(), verify_result);

    memcpy(h_vectors.hhY, h_vectors.temphY, d_coo_desc.rows * sizeof(DATA_TYPE));
    USPARSE_HIP_API_CHECK(
        hipMemcpy(dev_y, h_vectors.temphY, d_coo_desc.rows * sizeof(DATA_TYPE), hipMemcpyHostToDevice))
  }
};

#include "benchmark_usaprse_coomv.hpp"

// rocm
#ifdef __HIP_PLATFORM_HCC__
#include "benchmark_rocsparse_coomv.hpp"
#endif

// cuda
#ifndef __HIP_PLATFORM_HCC__
// TODO:
#endif

#endif // USPARSE_BENCHMARK_COO_SPMV
