//
// Created by luxu on 2022/12/08.
//
#ifndef SPGEMM_BENCHMARK_ROCSPARSE_CALL_HPP
#define SPGEMM_BENCHMARK_ROCSPARSE_CALL_HPP

#include <hipsparse.h>
#include <rocsparse.h>
#include "common.h"

// matrix turn by rocsparse csr2csc
template <typename I, typename T>
void rosparse_csr_matirx_turn(csr_mtx<I, T> &csr_matrix, I *&drow_ptr, I *&dcol_idx, T *&dval) {
  rocsparse_handle handle = NULL;
  ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

  I *dcsr_row_ptr, *dcsr_col_idx;
  T *dcsr_val;

  HIP_CHECK(hipMalloc((void **)&dcsr_row_ptr, (csr_matrix.rows + 1) * sizeof(I)))
  HIP_CHECK(hipMalloc((void **)&dcsr_col_idx, csr_matrix.nnz * sizeof(I)))
  HIP_CHECK(hipMalloc((void **)&dcsr_val, csr_matrix.nnz * sizeof(T)))
  HIP_CHECK(hipMemcpy(dcsr_row_ptr, csr_matrix.row_ptr, (csr_matrix.rows + 1) * sizeof(I), hipMemcpyHostToDevice))
  HIP_CHECK(hipMemcpy(dcsr_col_idx, csr_matrix.col_index, csr_matrix.nnz * sizeof(I), hipMemcpyHostToDevice))
  HIP_CHECK(hipMemcpy(dcsr_val, csr_matrix.values, csr_matrix.nnz * sizeof(T), hipMemcpyHostToDevice))

  HIP_CHECK(hipMalloc((void **)&drow_ptr, (csr_matrix.cols + 1) * sizeof(I)))
  HIP_CHECK(hipMalloc((void **)&dcol_idx, csr_matrix.nnz * sizeof(I)))
  HIP_CHECK(hipMalloc((void **)&dval, csr_matrix.nnz * sizeof(T)))

#if T == double
  size_t buffer_size;
  ROCSPARSE_CHECK(rocsparse_csr2csc_buffer_size(handle, csr_matrix.rows, csr_matrix.cols, csr_matrix.nnz, dcsr_row_ptr,
                                                dcsr_col_idx, rocsparse_action_numeric, &buffer_size))

  void *temp_buffer;
  HIP_CHECK(hipMalloc(&temp_buffer, buffer_size))
  ROCSPARSE_CHECK(rocsparse_dcsr2csc(handle, csr_matrix.rows, csr_matrix.cols, csr_matrix.nnz, dcsr_val, dcsr_row_ptr,
                                     dcsr_col_idx, dval, dcol_idx, drow_ptr, rocsparse_action_numeric,
                                     rocsparse_index_base_zero, temp_buffer))
  HIP_CHECK(hipFree(temp_buffer))
#elif T == float
  std::cout << "float rocsparse calls are not supported yet!" << std::endl;
  exit(0);
#endif

  ROCSPARSE_CHECK(rocsparse_destroy_handle(handle))

  HIP_CHECK(hipFree(dcsr_row_ptr))
  HIP_CHECK(hipFree(dcsr_col_idx))
  HIP_CHECK(hipFree(dcsr_val))
}

template <typename I, typename T>
size_t rocsparse_spgemm(const T alpha, const T beta, const bool opB, const size_t A_rows, const size_t A_cols,
                        const size_t A_nnz, I *const &drow_ptr_A, I *const &dcol_idx_A, T *const &dval_A,
                        const size_t B_nnz, I *const &drow_ptr_B, I *const &dcol_idx_B, T *const &dval_B,
                        I *&drow_ptr_C, I *&dcol_idx_C, T *&dval_C) {
  size_t B_cols = A_cols;
  if (opB) {
    B_cols = A_rows;
  }
  rocsparse_handle handle = NULL;
  ROCSPARSE_CHECK(rocsparse_create_handle(&handle));
  ROCSPARSE_CHECK(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

  rocsparse_mat_descr descr_A;
  rocsparse_mat_descr descr_B;
  rocsparse_mat_descr descr_C;
  rocsparse_mat_descr descr_D;

  ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr_A));
  ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr_B));
  ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr_C));
  ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr_D));

  rocsparse_mat_info info_C;
  ROCSPARSE_CHECK(rocsparse_create_mat_info(&info_C));

  ROCSPARSE_CHECK(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

  I D_nnz = 1;

  I *hrow_ptr_D = new idx_type[A_rows + 1];
  memset(hrow_ptr_D, 0, (A_rows + 1) * sizeof(I));
  hrow_ptr_D[A_rows] = 1;

  I hcol_idx_D = A_rows - 1;
  T hval_D = 0.0;

  I *drow_ptr_D;
  I *dcol_idx_D;
  T *dval_D;

  HIP_CHECK(hipMalloc((void **)&drow_ptr_D, (A_rows + 1) * sizeof(I)))
  HIP_CHECK(hipMalloc((void **)&dcol_idx_D, D_nnz * sizeof(I)))
  HIP_CHECK(hipMalloc((void **)&dval_D, D_nnz * sizeof(T)))

  HIP_CHECK(hipMemcpy(drow_ptr_D, hrow_ptr_D, (A_rows + 1) * sizeof(I), hipMemcpyHostToDevice))
  HIP_CHECK(hipMemcpy(dcol_idx_D, &hcol_idx_D, D_nnz * sizeof(I), hipMemcpyHostToDevice))
  HIP_CHECK(hipMemcpy(dval_D, &hval_D, D_nnz * sizeof(T), hipMemcpyHostToDevice))

  delete[] hrow_ptr_D;

  HIP_CHECK(hipMalloc((void **)&drow_ptr_C, sizeof(I) * (A_rows + 1)))

#if T == double
  size_t buffer_size;
  ROCSPARSE_CHECK(rocsparse_dcsrgemm_buffer_size(handle, rocsparse_operation_none, rocsparse_operation_none, A_rows,
                                                 B_cols, A_cols, &alpha, descr_A, A_nnz, drow_ptr_A, dcol_idx_A,
                                                 descr_B, B_nnz, drow_ptr_B, dcol_idx_B, &beta, descr_D, D_nnz,
                                                 drow_ptr_D, dcol_idx_D, info_C, &buffer_size))
  void *buffer;
  HIP_CHECK(hipMalloc(&buffer, buffer_size))

  rocsparse_int C_nnz = 0;

  ROCSPARSE_CHECK(rocsparse_csrgemm_nnz(handle, rocsparse_operation_none, rocsparse_operation_none, A_rows, B_cols,
                                        A_cols, descr_A, A_nnz, drow_ptr_A, dcol_idx_A, descr_B, B_nnz, drow_ptr_B,
                                        dcol_idx_B, descr_D, D_nnz, drow_ptr_D, dcol_idx_D, descr_C, drow_ptr_C, &C_nnz,
                                        info_C, buffer))

  HIP_CHECK(hipMalloc((void **)&dcol_idx_C, sizeof(I) * C_nnz))
  HIP_CHECK(hipMalloc((void **)&dval_C, sizeof(T) * C_nnz))

  ROCSPARSE_CHECK(rocsparse_dcsrgemm(handle, rocsparse_operation_none, rocsparse_operation_none, A_rows, B_cols, A_cols,
                                     &alpha, descr_A, A_nnz, dval_A, drow_ptr_A, dcol_idx_A, descr_B, B_nnz, dval_B,
                                     drow_ptr_B, dcol_idx_B, &beta, descr_D, D_nnz, dval_D, drow_ptr_D, dcol_idx_D,
                                     descr_C, dval_C, drow_ptr_C, dcol_idx_C, info_C, buffer))
  HIP_CHECK(hipFree(buffer))

#elif T == float
  std::cout << "float rocsparse calls are not supported yet!" << std::endl;
  exit(0);
#endif

  HIP_CHECK(hipFree(dval_D))
  HIP_CHECK(hipFree(dcol_idx_D))
  HIP_CHECK(hipFree(drow_ptr_D))

  return C_nnz;
}

#endif // SPGEMM_BENCHMARK_ROCSPARSE_CALL_HPP
