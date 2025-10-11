//
// Created by genshen on 2025/10/09.
//

#include <hip/hip_runtime_api.h>
#include <iostream>
#include <rocsparse/rocsparse.h>

#include "../utils/timer_utils.h"
#include "benchmark_csrmm.h"
#include "csr_spmm.hpp"

#define ROCSPARSE_CHECK(stat)                                                                                          \
  {                                                                                                                    \
    if (stat != rocsparse_status_success) {                                                                            \
      std::cerr << "Error: rocsparse error in line " << __LINE__ << std::endl;                                         \
    }                                                                                                                  \
  }

// https://github.com/ROCm/rocSPARSE/blob/develop_deprecated/clients/samples/example_spmm_csr.cpp
void RocSparseCsrMM::csr_spmm_impl(int trans, const dtype alpha, const dtype beta,
                                   const csr_desc<int, dtype> h_csr_desc, const csr_desc<int, dtype> d_csr_desc,
                                   const int M, const int K, const int N, dtype *B_dev, dtype *C_dev,
                                   BenchmarkTime *bmt) {
  hip::timer::event_timer calc_timer;
  
  rocsparse_handle handle;
  rocsparse_create_handle(&handle);
  
  rocsparse_mat_descr descr;
  ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr));
  
  calc_timer.start();
  ROCSPARSE_CHECK(rocsparse_dcsrmm(handle, rocsparse_operation_none, rocsparse_operation_none, M, N, K, d_csr_desc.nnz,
                                   &alpha, descr, d_csr_desc.values, d_csr_desc.row_ptr, d_csr_desc.col_index, B_dev, K,
                                   &beta, C_dev, M)); // B and C are column-order


  calc_timer.stop(true);

  double calc_time_cost = calc_timer.time_use;
  if (bmt != nullptr) {
    bmt->set_time(0., calc_time_cost, 0.0, 0.);
  }
}

bool RocSparseCsrMM::verify_beta_c() { return true; }

#undef ROCSPARSE_CHECK
