//
// Created by reget on 2021/11/16.
//

#ifndef SPMV_ACC_CSR_SPMV
#define SPMV_ACC_CSR_SPMV

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "csrmv_building_config.h"
#include "usparse_matrix.h"
#include "utils.hpp"
#include "utils/statistics_logger.h"
#include "verification.h"

template <typename INDEX_TYPE, typename DATA_TYPE> struct CsrSpMV {
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
  virtual void csr_spmv_impl(int trans, const DATA_TYPE alpha, const DATA_TYPE beta,
                             const csr_desc<INDEX_TYPE, DATA_TYPE> h_csr_desc,
                             const csr_desc<INDEX_TYPE, DATA_TYPE> d_csr_desc, const DATA_TYPE *x, DATA_TYPE *y,
                             BenchmarkTime *bmt) = 0;

  void test(std::string mtx_path, const std::string &strategy_name, enum usparse::usparse_operation operation,
            DATA_TYPE alpha, DATA_TYPE beta, csr_desc<INDEX_TYPE, DATA_TYPE> h_csr,
            csr_desc<INDEX_TYPE, DATA_TYPE> d_csr, host_vectors<DATA_TYPE> h_vectors, DATA_TYPE *&dev_x,
            DATA_TYPE *&dev_y) {
    BenchmarkTimeArray bmt_array;
    // warm up GPU
    for (int i = 0; i < 10; ++i) {
      // call sparse spmv
      HIP_CHECK(hipMemcpy(dev_y, h_vectors.temphY, d_csr.rows * sizeof(DATA_TYPE), hipMemcpyHostToDevice))
      try {
        csr_spmv_impl(operation, alpha, beta, h_csr, d_csr, dev_x, dev_y, nullptr);
      } catch (const std::runtime_error &error) {
        std::cout << "matrix name: " << mtx_path << ", strategy name: " << strategy_name << std::endl;
        std::cout << "error occur, return" << std::endl;
        std::cerr << error.what() << std::endl;
        hipDeviceSynchronize();
        memcpy(h_vectors.hhY, h_vectors.temphY, d_csr.rows * sizeof(DATA_TYPE));
        HIP_CHECK(hipMemcpy(dev_y, h_vectors.temphY, d_csr.rows * sizeof(DATA_TYPE), hipMemcpyHostToDevice))
        return;
      }
    }
    hipDeviceSynchronize();

    // execute device SpMV
    for (int i = 0; i < BENCHMARK_ARRAY_SIZE; i++) {
      BenchmarkTime bmt;
      HIP_CHECK(hipMemcpy(dev_y, h_vectors.temphY, d_csr.rows * sizeof(DATA_TYPE), hipMemcpyHostToDevice))
      hipDeviceSynchronize();
      csr_spmv_impl(operation, alpha, beta, h_csr, d_csr, dev_x, dev_y, &bmt);
      hipDeviceSynchronize();
      bmt_array.append(bmt);
    }

    // device result check
    HIP_CHECK(hipMemcpy(dev_y, h_vectors.temphY, h_csr.rows * sizeof(DATA_TYPE), hipMemcpyHostToDevice))
    csr_spmv_impl(operation, alpha, beta, h_csr, d_csr, dev_x, dev_y, nullptr);
    hipDeviceSynchronize();
    HIP_CHECK(hipMemcpy(h_vectors.hY, dev_y, d_csr.rows * sizeof(DATA_TYPE), hipMemcpyDeviceToHost));

    // host side verification
    if (verify_beta_y()) {
      // y = alpha*A*x + beta*y
      host_csr_spmv(alpha, beta, h_csr.values, h_csr.row_ptr, h_csr.col_index, h_csr.rows, h_csr.cols, h_csr.nnz,
                    h_vectors.hX, h_vectors.hhY);
    } else {
      // y = A*x
      host_csr_spmv(h_csr.values, h_csr.row_ptr, h_csr.col_index, h_csr.rows, h_csr.cols, h_csr.nnz, h_vectors.hX,
                    h_vectors.hhY);
    }

    const VerifyResult<INDEX_TYPE, DATA_TYPE> verify_result = verify_y(h_vectors.hY, h_vectors.hhY, h_csr.rows);
    statistics::print_statistics<DATA_TYPE>(mtx_path, strategy_name, h_csr.rows, h_csr.cols, h_csr.nnz,
                                            bmt_array.get_mid_time(), verify_result);

    memcpy(h_vectors.hhY, h_vectors.temphY, d_csr.rows * sizeof(DATA_TYPE));
    HIP_CHECK(hipMemcpy(dev_y, h_vectors.temphY, d_csr.rows * sizeof(DATA_TYPE), hipMemcpyHostToDevice))
  }
};

#include "benchmark_spmv_acc.hpp"

// rocm
#ifdef __HIP_PLATFORM_HCC__
#ifdef UBENCH_CONFIG_ENABLE_HOLA_ALGO
#include "benchmark_hola_hip.hpp"
#endif // UBENCH_CONFIG_ENABLE_HOLA_ALGO
#include "benchmark_rocsparse.hpp"
#endif

// cuda
#ifndef __HIP_PLATFORM_HCC__
#include "benchmark_cub.hpp"
#include "benchmark_cusparse.hpp"
#ifdef UBENCH_CONFIG_ENABLE_HOLA_ALGO
#include "benchmark_hola_cuda.hpp"
#endif // UBENCH_CONFIG_ENABLE_HOLA_ALGO
#include "benchmark_acsr.hpp"
#include "benchmark_merge_path.hpp"
#endif

#endif
