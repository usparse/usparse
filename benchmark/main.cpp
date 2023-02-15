//
// Created by luxu on 2022/12/08.
//
#include <hip/hip_runtime.h>
#include <iostream>
#include <stdlib.h>
#include <sys/time.h>
#include <vector>
#include "api/spgemm_hip_api.h"
#include "cli/csr_binary_reader.hpp"
#include "cli/matrix_market_reader.hpp"
#include "cli/sparse_format.h"
#include "common.h"
#include "memory_copy.hpp"
#include "vertify.hpp"
#include "count.hpp"

int main(int argc, char **argv) {
  std::cout << "==========================USTB-SpGEMM=============================" << std::endl;
  std::cout << "Instructions: Calculate A*A A*AT or A*randomB" << std::endl;
  std::cout << "[1] Read matrix from mtx file, input args should be: " << std::endl;
  std::cout << "    mtx file path, opB(0/1), mtx, " << std::endl;
  std::cout << "[2] Read matrix from bin file, input args should be: " << std::endl;
  std::cout << "    bin file path, opB(0/1), bin" << std::endl;
  std::cout << "[3] Read matrix from file & creat random matrixB, input args should be: " << std::endl;
  std::cout << "    file path, opB(2), sparsityB, bin/mtx" << std::endl;
  std::cout << "==================================================================" << std::endl;

  // Set random alpha
  srand((unsigned)time(NULL));
  // data_type alpha = rand() / 10000000.0;
  data_type alpha = 1.0;
  data_type beta = 0.0;
  data_type sparsityB = 0.0;
  idx_type opB = 2;
  const int DCU_ROUND = 1;

  char filename[1024];
  strcpy(filename, argv[1]);

  opB = atoi(argv[2]);

  char filetype[128];

  if (opB == 2) {
    sparsityB = atof(argv[3]);
    strcpy(filetype, argv[4]);
  } else {
    strcpy(filetype, argv[3]);
  }

  // Host variable declaration
  csr_mtx<idx_type, data_type> matrix_in, matrix_c, compare_matrix_c;

  std::cout << "Preparing Host Memory for matrixA & matrixB ... " << std::endl;
  if (strcmp(filetype, "mtx") == 0) {
    matrix_market_reader<idx_type, data_type> mm_reader;
    matrix_market<idx_type, data_type> csr_mm_reader = mm_reader.load_mat(filename);
    matrix_in = csr_mm_reader.to_csr();
  } else if (strcmp(filetype, "bin") == 0) {
    csr_binary_reader<idx_type, data_type> csr_bin_reader;
    csr_bin_reader.load_mat(filename);
    csr_bin_reader.close_stream();

    matrix_in.rows = csr_bin_reader.rows();
    matrix_in.cols = csr_bin_reader.cols();
    matrix_in.nnz = csr_bin_reader.nnz();
    // just reuse memory in file reading step.
    csr_bin_reader.as_raw_ptr(matrix_in.values, matrix_in.col_index, matrix_in.row_ptr);
  }

  std::cout << "Host Data Created Successfully!" << std::endl;
  std::cout << "==================================================================" << std::endl;
  std::cout << "Input Matrix Data Info:" << std::endl;
  std::cout << "alpha=" << alpha << ", beta=" << beta << std::endl;
  std::cout << "rows of matrix_in=" << matrix_in.rows << std::endl;
  std::cout << "cols of matrix_in=" << matrix_in.cols << std::endl;
  std::cout << "nnz of matrix_in=" << matrix_in.nnz << std::endl;
  std::cout << "==================================================================" << std::endl;

  // Set Device Id for the following Calculation
  hipSetDevice(0);

  // Device variable declaration
  idx_type *drow_ptr_A, *drow_ptr_B, *drow_ptr_C, *compare_drow_ptr_C;
  idx_type *dcol_idx_A, *dcol_idx_B, *dcol_idx_C, *compare_dcol_idx_C;
  data_type *dval_A, *dval_B, *dval_C, *compare_dval_C;

  create_deivce_data<idx_type, data_type>(matrix_in, drow_ptr_A, dcol_idx_A, dval_A, opB, sparsityB, drow_ptr_B, dcol_idx_B,
                                          dval_B, drow_ptr_C);

  std::cout << "Device Data Created Successfully!" << std::endl;
  std::cout << "==================================================================" << std::endl;

  size_t C_nnz = 0;

  std::cout << "Runing USTB-SpGEMM Warmup ... " << std::endl;

  // Start timer kernel
  common::timer[ustb_spgemm_warmup_timer].start();

  // Execute Device spgemm
  sparse_gemm_hip<data_type, idx_type>(hip_operation_none, hip_operation_none, alpha, beta, matrix_in.rows,
                                       matrix_in.rows, matrix_in.cols, matrix_in.nnz, drow_ptr_A, dcol_idx_A, dval_A,
                                       matrix_in.nnz, drow_ptr_B, dcol_idx_B, dval_B, &C_nnz, drow_ptr_C, &dcol_idx_C,
                                       &dval_C);

  hipDeviceSynchronize();

  // Stop timer kernel
  common::timer[ustb_spgemm_warmup_timer].stop();

  copy_results_from_device<idx_type, data_type>(matrix_c, opB, matrix_in.rows, matrix_in.cols, C_nnz, drow_ptr_C,
                                                dcol_idx_C, dval_C);

  HIP_CHECK(hipFree(dcol_idx_C))

  std::cout << "USTB-SpGEMM warmup non zero num of matrix C:" << C_nnz << std::endl;
  std::cout << "USTB-SpGEMM Warmup finished! elapsed time:" << common::timer[ustb_spgemm_warmup_timer].time_use.count() << "(us)" << std::endl;
  std::cout << "==================================================================" << std::endl;

  std::cout << "Runing USTB-SpGEMM ... " << std::endl;

  double ustb_spgemm_timer_mean = 0.0;
  double ustb_spgemm_symbolic_timer_mean = 0.0;
  double ustb_spgemm_numeric_timer_mean = 0.0;

  for (int i = 0; i < DCU_ROUND; i++) {
    // Start timer kernel
    common::timer[ustb_spgemm_timer].start();

    // Execute Device spgemm
    sparse_gemm_hip<data_type, idx_type>(hip_operation_none, hip_operation_none, alpha, beta, matrix_in.rows,
                                         matrix_in.rows, matrix_in.cols, matrix_in.nnz, drow_ptr_A, dcol_idx_A, dval_A,
                                         matrix_in.nnz, drow_ptr_B, dcol_idx_B, dval_B, &C_nnz, drow_ptr_C, &dcol_idx_C,
                                         &dval_C);

    hipDeviceSynchronize();

    // Stop timer kernel
    common::timer[ustb_spgemm_timer].stop();

    HIP_CHECK(hipFree(dcol_idx_C))

    ustb_spgemm_timer_mean += common::timer[ustb_spgemm_timer].time_use.count();
    ustb_spgemm_symbolic_timer_mean += common::timer[ustb_spgemm_symbolic_timer].time_use.count();
    ustb_spgemm_numeric_timer_mean += common::timer[ustb_spgemm_numeric_timer].time_use.count();

    std::cout << "Round: " << i << "  elapsed time:" << common::timer[ustb_spgemm_timer].time_use.count() << "(us)"
              << std::endl;
  }
  std::cout << "==================================================================" << std::endl;

  HIP_CHECK(hipFree(drow_ptr_C))

  if (DCU_ROUND > 0) {
    ustb_spgemm_timer_mean = ustb_spgemm_timer_mean / (DCU_ROUND * 1.0);
    ustb_spgemm_symbolic_timer_mean = ustb_spgemm_symbolic_timer_mean / (DCU_ROUND * 1.0);
    ustb_spgemm_numeric_timer_mean = ustb_spgemm_numeric_timer_mean / (DCU_ROUND * 1.0);
  }

  std::cout << "USTB-SpGEMM Test Execution Result:" << std::endl;
  std::cout << "USTB-SpGEMM calculate non zero num of matrix C:" << C_nnz << std::endl;
  std::cout << "USTB-SpGEMM calculation finished! Total elapsed time (mean):" << ustb_spgemm_timer_mean << "(us)" << std::endl;
  std::cout << "USTB-SpGEMM calculation finished! Symbolic elapsed time (mean):" << ustb_spgemm_symbolic_timer_mean << "(us)" << std::endl;
  std::cout << "USTB-SpGEMM calculation finished! Numeric elapsed time (mean):" << ustb_spgemm_numeric_timer_mean << "(us)" << std::endl;
  std::cout << "==================================================================" << std::endl;

  size_t compare_C_nnz = 0;

  std::cout << "Runing Rocsparse-SpGEMM Warmup ... " << std::endl;

  // Start timer kernel
  common::timer[rocsparse_spgemm_warmup_timer].start();

  // Execute rocsparse spgemm
  compare_C_nnz = rocsparse_spgemm<idx_type, data_type>(
      alpha, beta, opB, matrix_in.rows, matrix_in.cols, matrix_in.nnz, drow_ptr_A, dcol_idx_A, dval_A, matrix_in.nnz,
      drow_ptr_B, dcol_idx_B, dval_B, compare_drow_ptr_C, compare_dcol_idx_C, compare_dval_C);

  hipDeviceSynchronize();

  // Stop timer kernel
  common::timer[rocsparse_spgemm_warmup_timer].stop();

  copy_results_from_device<idx_type, data_type>(compare_matrix_c, opB, matrix_in.rows, matrix_in.cols, compare_C_nnz,
                                                compare_drow_ptr_C, compare_dcol_idx_C, compare_dval_C);

  HIP_CHECK(hipFree(compare_drow_ptr_C))
  HIP_CHECK(hipFree(compare_dcol_idx_C))
  HIP_CHECK(hipFree(compare_dval_C))

  std::cout << "Rocsparse-SpGEMM warmup non zero num of matrix C:" << compare_C_nnz << std::endl;
  std::cout << "Rocsparse-SpGEMM Warmup finished! elapsed time:" << common::timer[rocsparse_spgemm_warmup_timer].time_use.count() << "(us)" << std::endl;
  std::cout << "==================================================================" << std::endl;

  std::cout << "Runing Rocsparse-SpGEMM ... " << std::endl;

  double rocsparse_spgemm_timer_mean = 0.0;
  double rocsparse_spgemm_symbolic_timer_mean = 0.0;
  double rocsparse_spgemm_numeric_timer_mean = 0.0;

  for (int i = 0; i < DCU_ROUND; i++) {

    // Execute Device spgemm
    compare_C_nnz = rocsparse_spgemm<idx_type, data_type>(
        alpha, beta, opB, matrix_in.rows, matrix_in.cols, matrix_in.nnz, drow_ptr_A, dcol_idx_A, dval_A, matrix_in.nnz,
        drow_ptr_B, dcol_idx_B, dval_B, compare_drow_ptr_C, compare_dcol_idx_C, compare_dval_C);

    hipDeviceSynchronize();

    HIP_CHECK(hipFree(compare_drow_ptr_C))
    HIP_CHECK(hipFree(compare_dcol_idx_C))
    HIP_CHECK(hipFree(compare_dval_C))

    rocsparse_spgemm_timer_mean += common::timer[rocsparse_spgemm_timer].time_use.count();
    rocsparse_spgemm_symbolic_timer_mean += common::timer[rocsparse_spgemm_symbolic_timer].time_use.count();
    rocsparse_spgemm_numeric_timer_mean += common::timer[rocsparse_spgemm_numeric_timer].time_use.count();

    std::cout << "Round: " << i << "  elapsed time:" << common::timer[rocsparse_spgemm_timer].time_use.count() << "(us)"
              << std::endl;
  }
  std::cout << "==================================================================" << std::endl;


  if (DCU_ROUND > 0) {
    rocsparse_spgemm_timer_mean = rocsparse_spgemm_timer_mean / (DCU_ROUND * 1.0);
    rocsparse_spgemm_symbolic_timer_mean = rocsparse_spgemm_symbolic_timer_mean / (DCU_ROUND * 1.0);
    rocsparse_spgemm_numeric_timer_mean = rocsparse_spgemm_numeric_timer_mean / (DCU_ROUND * 1.0);
  }

  std::cout << "Rocsparse-SpGEMM Test Execution Result:" << std::endl;
  std::cout << "Rocsparse-SpGEMM calculate non zero num of matrix C:" << compare_C_nnz << std::endl;
  std::cout << "Rocsparse-SpGEMM calculation finished! Total elapsed time (mean):" << rocsparse_spgemm_timer_mean << "(us)" << std::endl;
  std::cout << "Rocsparse-SpGEMM calculation finished! Symbolic elapsed time (mean):" << rocsparse_spgemm_symbolic_timer_mean << "(us)" << std::endl;
  std::cout << "Rocsparse-SpGEMM calculation finished! Numeric elapsed time (mean):" << rocsparse_spgemm_numeric_timer_mean << "(us)" << std::endl;
  std::cout << "==================================================================" << std::endl;

  vertify<idx_type, data_type>(matrix_c, compare_matrix_c);
  std::cout << "==================================================================" << std::endl;

  std::cout << "===========================     COUNT   ==========================" << std::endl;
  int symbolic_operations = symbolic_num_of_operations<idx_type>(matrix_in.cols, matrix_in.nnz, dcol_idx_A, drow_ptr_B);
  std::cout << "Number of symbol phase operations:" << symbolic_operations << std::endl;
  int numeric_add_counts =  numeric_nums_of_add<idx_type>(matrix_in.cols, matrix_in.nnz, dcol_idx_A, drow_ptr_B);
  std::cout << "Number of additions in numeric phase:" << numeric_add_counts << std::endl;
  int numeric_mul_counts =  numeric_nums_of_mul<idx_type>(matrix_in.cols, matrix_in.nnz, dcol_idx_A, drow_ptr_B, compare_C_nnz);
  std::cout << "Number of multiplications in numeric phase:" << numeric_mul_counts << std::endl;
  return 0;
}
