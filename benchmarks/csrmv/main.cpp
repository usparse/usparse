//
// Created by genshen on 2021/11/15.
//

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "usparse_matrix.h"
#include "usparse_types.h"

#include "../utils/benchmark_main.hpp"
#include "benchmark_merge_path.hpp"
#include "matrix_format/csr_binary_reader.hpp"
#include "matrix_format/csr_mtx_reader.hpp"
#include "matrix_format/matrix_market_reader.hpp"
#include "matrix_format/sparse_format.h"
#include "utils.hpp"
#include "utils/statistics_logger.h"

#include "benchmark_config.h"
#include "csr_spmv.hpp"
#include "csrmv_building_config.h"

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

#define SPMV_BENCHMARK(instance, name, flag)                                                                           \
  {                                                                                                                    \
    instance<INDEX_TYPE, DATA_TYPE> ins;                                                                               \
    if (flag) {                                                                                                        \
      ins.test(mtx_path, name, operation, alpha, beta, h_csr, d_csr, h_vectors, dev_x, dev_y);                         \
    }                                                                                                                  \
  }

template <typename INDEX_TYPE, typename DATA_TYPE>
void test_spmv(std::string mtx_path, csr_mtx<INDEX_TYPE, DATA_TYPE> h_csr, host_vectors<DATA_TYPE> h_vectors) {
  USPARSE_HIP_API_CHECK(hipSetDevice(0));

  DATA_TYPE *dev_x, *dev_y;
  csr_mtx<INDEX_TYPE, DATA_TYPE> d_csr = create_device_data(h_csr, h_vectors.hX, h_vectors.temphY, dev_x, dev_y);
  // set parameters
  enum usparse::usparse_operation operation = usparse::usparse_operation::operation_none;
  DATA_TYPE alpha = 1.0;
  DATA_TYPE beta = 1.0;

  statistics::print_statistics_header();

  // spmv-acc
  // SPMV_BENCHMARK(SpMVAccDefault, "spmv-acc-default", ENABLE_SPMV_ACC_DEFAULT);
  SPMV_BENCHMARK(SpMVAccAdaptive, "spmv-acc-adaptive", ENABLE_SPMV_ACC_ADAPTIVE);
  SPMV_BENCHMARK(SpMVAccFlat, "spmv-acc-flat", ENABLE_SPMV_ACC_FLAT);
  SPMV_BENCHMARK(SpMVAccFlatV2, "spmv-acc-flat-v2", ENABLE_SPMV_ACC_FLAT);
  SPMV_BENCHMARK(SpMVAccLight, "spmv-acc-light", ENABLE_SPMV_ACC_LIGHT);
  if constexpr (std::is_same<DATA_TYPE, usparse::usparse_fp32>::value) {
    // line and thread_row only support fp32 now
    SPMV_BENCHMARK(SpMVAccLine, "spmv-acc-line", ENABLE_SPMV_ACC_LINE);
    SPMV_BENCHMARK(SpMVAccThreadRow, "spmv-acc-thread-row", ENABLE_SPMV_ACC_THREAD_ROW);
  }
  SPMV_BENCHMARK(SpMVAccVecRow, "spmv-acc-vector-row", ENABLE_SPMV_ACC_VECTOR_ROW);
  SPMV_BENCHMARK(SpMVAccLineEnhance, "spmv-acc-line-enhance", ENABLE_SPMV_ACC_LE_ROW);
  SPMV_BENCHMARK(SpMVAccAdaptivePlus, "spmv-acc-adaptive-plus", ENABLE_SPMV_ACC_ADAPTIVE_PLUS);
  SPMV_BENCHMARK(SpMVAccFlatSegSum, "spmv-acc-flat-seg-sum", ENABLE_SPMV_ACC_FLAT_SEG_SUM);
  SPMV_BENCHMARK(SpMVAccFlatSegSumV2, "spmv-acc-flat-seg-sum-v2", ENABLE_SPMV_ACC_FLAT_SEG_SUM_V2);

#ifdef __USPARSE_BUILD_BACKEND_ROCM__
  // rocsparse
  SPMV_BENCHMARK(RocSparseVecRow, "rocSparse-vector-row", ENABLE_ROC_VECTOR_ROW);
  SPMV_BENCHMARK(RocSparseAdaptive, "rocSparse-adaptive", ENABLE_ROC_ADAPTIVE);
#ifdef UBENCH_CONFIG_ENABLE_HOLA_ALGO
  // hola
  SPMV_BENCHMARK(HolaHipSpMV, "hip-hola", ENABLE_HIP_HOLA);
#endif
#endif

#ifdef __USPARSE_BUILD_BACKEND_CUDA__
  // cusparse
  SPMV_BENCHMARK(CuSparseGeneral, "cuSparse", ENABLE_CU_SPARSE);
  // cub
  SPMV_BENCHMARK(CubDeviceSpMV, "cub", ENABLE_CUB);
#ifdef UBENCH_CONFIG_ENABLE_HOLA_ALGO
  // hola
  SPMV_BENCHMARK(HolaSpMV, "hola", ENABLE_HOLA);
#endif
  // merge path
  SPMV_BENCHMARK(MergePathSingleBlockUpdateSpMV, "merge-path-single-block-update", ENABLE_MERGE_PATH)
  SPMV_BENCHMARK(MergePathLookBackUpdateSpMV, "merge-path-look-back-update", ENABLE_MERGE_PATH)
  // ACSR
  SPMV_BENCHMARK(ACSRSpMV, "acsr", ENABLE_ACSR);
#endif

#ifdef __USPARSE_BUILD_BACKEND_MUSA__
  SPMV_BENCHMARK(MuSparse, "musparse", ENABLE_MUSPARSE);
#endif

  destroy_device_data(d_csr, dev_x, dev_y);
}
