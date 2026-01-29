//
// Created by genshen on 2025/10/09.
//

#include <hip/hip_runtime_api.h>
#include <iostream>
#include <rocsparse/rocsparse.h>

#include "../utils/timer_utils.h"
#include "benchmark_coomm.h"
#include "coo_spmm.hpp"

#define ROCSPARSE_CHECK(stat)                                                                                          \
  {                                                                                                                    \
    if (stat != rocsparse_status_success) {                                                                            \
      std::cerr << "Error: rocsparse error in line " << __LINE__ << std::endl;                                         \
    }                                                                                                                  \
  }

// https://github.com/ROCm/rocSPARSE/blob/develop_deprecated/clients/samples/example_spmm_coo.cpp
template <typename I, typename T>
void RocSparseCooMM<I, T>::coo_spmm_impl(int trans, const T alpha, const T beta,
                                         const usparse::coo_desc<I, T> h_coo_desc,
                                         const usparse::coo_desc<I, T> d_coo_desc, const I M, const I K, const I N,
                                         T *B_dev, T *C_dev, BenchmarkTime *bmt) {
  hip::timer::event_timer pre_time, calc_timer, destroy_time;

  rocsparse_handle handle;
  rocsparse_create_handle(&handle);

  T halpha = alpha;
  T hbeta = beta;

  // Types
  rocsparse_indextype itype = rocsparse_indextype_i32;
  rocsparse_datatype ttype = rocsparse_datatype_f64_r;

  // Create descriptors
  rocsparse_spmat_descr A;
  rocsparse_dnmat_descr B;
  rocsparse_dnmat_descr C;

  ROCSPARSE_CHECK(rocsparse_create_coo_descr(&A, M, K, d_coo_desc.nnz, const_cast<I *>(d_coo_desc.row_index),
                                             const_cast<I *>(d_coo_desc.col_index), const_cast<T *>(d_coo_desc.values),
                                             itype, rocsparse_index_base_zero, ttype));
  ROCSPARSE_CHECK(rocsparse_create_dnmat_descr(&B, K, N, K, B_dev, ttype, rocsparse_order_column));
  ROCSPARSE_CHECK(rocsparse_create_dnmat_descr(&C, M, N, M, C_dev, ttype, rocsparse_order_column));

  pre_time.start();
  // Query for buffer size
  size_t buffer_size;
  ROCSPARSE_CHECK(rocsparse_spmm(handle, rocsparse_operation_none, rocsparse_operation_none, &halpha, A, B, &hbeta, C,
                                 ttype, rocsparse_spmm_alg_default, rocsparse_spmm_stage_buffer_size, &buffer_size,
                                 nullptr));
  void *temp_buffer;
  USPARSE_HIP_API_CHECK(hipMalloc(&temp_buffer, buffer_size));

  // preprocess.
  ROCSPARSE_CHECK(rocsparse_spmm(handle, rocsparse_operation_none, rocsparse_operation_none, &halpha, A, B, &hbeta, C,
                                 ttype, rocsparse_spmm_alg_default, rocsparse_spmm_stage_preprocess, &buffer_size,
                                 temp_buffer));
  pre_time.stop(true);

  // COO matrix matrix multiplication
  calc_timer.start();
  // Call rocsparse spmm
  ROCSPARSE_CHECK(rocsparse_spmm(handle, rocsparse_operation_none, rocsparse_operation_none, &halpha, A, B, &hbeta, C,
                                 ttype, rocsparse_spmm_alg_default, rocsparse_spmm_stage_compute, &buffer_size,
                                 temp_buffer));
  calc_timer.stop(true);

  destroy_time.start();
  USPARSE_HIP_API_CHECK(hipFree(temp_buffer));
  ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(A));
  ROCSPARSE_CHECK(rocsparse_destroy_dnmat_descr(B));
  ROCSPARSE_CHECK(rocsparse_destroy_dnmat_descr(C));
  destroy_time.stop();

  if (bmt != nullptr) {
    bmt->set_time(pre_time.time_use, calc_timer.time_use, 0.0, destroy_time.time_use);
  }
}

template <typename I, typename T> bool RocSparseCooMM<I, T>::verify_beta_c() { return true; }

template <typename I, typename T> bool RocSparseCooMM<I, T>::trans_B_C() { return true; }

template class RocSparseCooMM<int, float>;

template class RocSparseCooMM<int, double>;

#undef ROCSPARSE_CHECK
