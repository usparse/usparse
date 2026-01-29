//
// Created by luxu on 2022/12/08.
//

#ifndef SPGEMM_BENCHMARK_MEMORY_MANAGEMENT_HPP
#define SPGEMM_BENCHMARK_MEMORY_MANAGEMENT_HPP

#include "rocsparse_call.hpp"
#include "creat_random_matrix.hpp"

template <typename I, typename T>
void create_deivce_data(csr_mtx<I, T> &matrix, I *&drow_ptr_A, I *&dcol_idx_A, T *&dval_A, I opB, T sparsityB,I *&drow_ptr_B,
                        I *&dcol_idx_B, T *&dval_B, I *&drow_ptr_C) {
  std::cout << "Preparing Device Memory for matrixA ..." << std::endl;
  USPARSE_HIP_API_CHECK(hipMalloc((void **)&drow_ptr_A, (matrix.rows + 1) * sizeof(I)))
  USPARSE_HIP_API_CHECK(hipMalloc((void **)&dcol_idx_A, matrix.nnz * sizeof(I)))
  USPARSE_HIP_API_CHECK(hipMalloc((void **)&dval_A, matrix.nnz * sizeof(T)))
  USPARSE_HIP_API_CHECK(hipMemcpy(drow_ptr_A, matrix.row_ptr, (matrix.rows + 1) * sizeof(I), hipMemcpyHostToDevice))
  USPARSE_HIP_API_CHECK(hipMemcpy(dcol_idx_A, matrix.col_index, matrix.nnz * sizeof(I), hipMemcpyHostToDevice))
  USPARSE_HIP_API_CHECK(hipMemcpy(dval_A, matrix.values, matrix.nnz * sizeof(T), hipMemcpyHostToDevice))

  std::cout << "Preparing Device Memory for matrixB ..." << std::endl;
  if (opB == 0) {
    std::cout << "MatrixB is equal to matrixA" << std::endl;
    USPARSE_HIP_API_CHECK(hipMalloc((void **)&drow_ptr_B, (matrix.rows + 1) * sizeof(I)))
    USPARSE_HIP_API_CHECK(hipMalloc((void **)&dcol_idx_B, matrix.nnz * sizeof(I)))
    USPARSE_HIP_API_CHECK(hipMalloc((void **)&dval_B, matrix.nnz * sizeof(T)))
    USPARSE_HIP_API_CHECK(hipMemcpy(drow_ptr_B, matrix.row_ptr, (matrix.rows + 1) * sizeof(I), hipMemcpyHostToDevice))
    USPARSE_HIP_API_CHECK(hipMemcpy(dcol_idx_B, matrix.col_index, matrix.nnz * sizeof(I), hipMemcpyHostToDevice))
    USPARSE_HIP_API_CHECK(hipMemcpy(dval_B, matrix.values, matrix.nnz * sizeof(T), hipMemcpyHostToDevice))
  } else if(opB == 1){
    std::cout << "MatrixB is transposition of matrixA" << std::endl;
    rosparse_csr_matirx_turn<I, T>(matrix, drow_ptr_B, dcol_idx_B, dval_B);
  } else if(opB == 2){
    std::cout << "MatrixB is generated randomly" << std::endl;
    create_random_matrix<I, T>(matrix, drow_ptr_B, dcol_idx_B, dval_B, sparsityB);
  } else {
    std::cout << "There is no opB option" << std::endl;
    exit(0);
  }

  std::cout << "Preparing Device Memory for matrixC's row_ptr ..." << std::endl;
  USPARSE_HIP_API_CHECK(hipMalloc((void **)&drow_ptr_C, (matrix.rows + 1) * sizeof(I)))
}

template <typename I, typename T>
void copy_results_from_device(csr_mtx<I, T> &matrix, const bool opB, const size_t A_rows, const size_t A_cols,
                              const size_t C_nnz, I *const &drow_ptr, I *const &dcol_idx, T *const &dval) {
  size_t C_cols = A_cols;
  if (opB) {
    C_cols = A_rows;
  }
  matrix.alloc(A_rows, A_cols, C_nnz);
  USPARSE_HIP_API_CHECK(hipMemcpy(matrix.row_ptr, drow_ptr, (A_rows + 1) * sizeof(I), hipMemcpyDeviceToHost))
  USPARSE_HIP_API_CHECK(hipMemcpy(matrix.col_index, dcol_idx, C_nnz * sizeof(I), hipMemcpyDeviceToHost))
  USPARSE_HIP_API_CHECK(hipMemcpy(matrix.values, dval, C_nnz * sizeof(T), hipMemcpyDeviceToHost))
}

#endif // SPGEMM_BENCHMARK_MEMORY_MANAGEMENT_HPP
