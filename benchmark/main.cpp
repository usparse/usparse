//
// Created by luxu on 2022/12/08.
//
#include <iostream>
#include <stdlib.h>
#include <sys/time.h>
#include <vector>
#include <hip/hip_runtime.h>
#include "common.h"
#include "api/spgemm_hip_api.h"
#include "cli/sparse_format.h"
#include "cli/matrix_market_reader.hpp"
#include "memory_copy.hpp"


int main(int argc, char **argv){
  std::cout << "==========================USTB-SpGEMM=============================" << std::endl;
  std::cout << "Instructions: Calculate A*AT" << std::endl;
  std::cout << "[1] Read matrix from mtx file, input args should be: " << std::endl;
  std::cout << "    mtx file path, opB(0/1), mtx, " << std::endl;
  std::cout << "[2] Read matrix from bin file, input args should be: " << std::endl;
  std::cout << "    bin file path, opB(0/1), bin" << std::endl;
  std::cout << "==================================================================" << std::endl;

  // Set random alpha
  srand((unsigned)time(NULL));
  data_type alpha = rand() / 10000000.0;
  data_type beta = 0.0;
  bool opB = false;

  char filename[1024];
  strcpy(filename, argv[1]);

  int operator_B = atoi(argv[2]);
  if(operator_B == 1){
    opB = true;
  }


  char filetype[128];
  strcpy(filetype, argv[3]);

  // Host variable declaration
  csr_mtx<idx_type,data_type> matrix_in, matrix_c, compare_matrix_c;

  std::cout << "Preparing Host Memory for matrixA & matrixB ... " << std::endl;
  if(strcmp(filetype,"mtx") == 0){
    matrix_market_reader<idx_type,data_type> mtr;
    matrix_market<idx_type, data_type> csr_mm=mtr.load_mat(filename);
    matrix_in=csr_mm.to_csr();
  } else if(strcmp(filetype,"bin") == 0){
    std::cout<<"Reading binary files is not supported!"<<std::endl;
    return 0;
  }

  std::cout<<"Host Data Created Successfully!"<<std::endl;
  std::cout << "==================================================================" << std::endl;
  std::cout << "Input Matrix Data Info:" << std::endl;
  std::cout << "alpha = " << alpha << ", beta = " << beta << std::endl;
  std::cout << "rows of matrix_in=" << matrix_in.rows << std::endl;
  std::cout << "cols of matrix_in=" << matrix_in.cols << std::endl;
  std::cout << "nnz of matrix_in=" << matrix_in.nnz << std::endl;
  std::cout << "==================================================================" << std::endl;

  // Set Device Id for the following Calculation
  hipSetDevice(0);

  // Device variable declaration
  idx_type *drow_ptr_A, *drow_ptr_B, *drow_ptr_C;
  idx_type *dcol_idx_A, *dcol_idx_B, *dcol_idx_C;
  data_type *dval_A, *dval_B, *dval_C;

  create_deivce_data<idx_type,data_type>(matrix_in,drow_ptr_A,dcol_idx_A,dval_A,opB,drow_ptr_B,dcol_idx_B,dval_B,drow_ptr_C);

  std::cout<<"Device Data Created Successfully!"<<std::endl;
  std::cout << "==================================================================" << std::endl;

  size_t C_nnz = 0;

  std::cout << "Runing Warmup ... " << std::endl;

  // Start timer kernel
  // common::timer[ustb_spgemm_warmup_timer].start();

  // Execute Device spgemm
  sparse_gemm_hip<data_type, idx_type>(hip_operation_none, hip_operation_none, alpha, beta, matrix_in.rows, matrix_in.rows, matrix_in.cols, matrix_in.nnz, drow_ptr_A, dcol_idx_A,
                     dval_A, matrix_in.nnz, drow_ptr_B, dcol_idx_B, dval_B, &C_nnz, drow_ptr_C,
                     &dcol_idx_C, &dval_C);

  hipDeviceSynchronize();

  // Stop timer kernel
  // common::timer[ustb_spgemm_warmup_timer].stop();

  HIP_CHECK(hipFree(dcol_idx_C))

  std::cout << "non zero num of matrix C : " << C_nnz << std::endl;
  // std::cout << "Warmup finished!  elapsed time:" << common::timer[ustb_spgemm_warmup_timer].time_use.count() << "(us)" << std::endl;
  std::cout << "==================================================================" << std::endl;


  return 0;
}
