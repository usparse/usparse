#include "sparseMatrix.hpp"
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

#ifdef __USPARSE_BUILD_BACKEND_ROCM__

#include <hipblas.h>
#include <hipsparse.h>

double Verification(SpM<double> *d_A, double *x, double *b, double *r0, int N) {
  hipsparseHandle_t spHandle;
  hipsparseCreate(&spHandle);

  hipblasHandle_t blasHandle;
  hipblasCreate(&blasHandle);

  hipsparseSpMatDescr_t matA;
  hipsparseCreateCsr(&matA, N, N, d_A->nnz, d_A->rows, d_A->cols, d_A->vals, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
                     HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F);

  hipsparseDnVecDescr_t vecX, vecY;
  hipsparseCreateDnVec(&vecX, N, x, HIP_R_64F);
  hipsparseCreateDnVec(&vecY, N, r0, HIP_R_64F);

  double alpha = 1.0, beta = 0.0;
  size_t bufferSize = 0;
  void *dBuffer = nullptr;

  hipsparseSpMV_bufferSize(spHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, HIP_R_64F,
                           HIPSPARSE_SPMV_ALG_DEFAULT, &bufferSize);

  hipMalloc(&dBuffer, bufferSize);

  hipsparseSpMV(spHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, HIP_R_64F,
                HIPSPARSE_SPMV_ALG_DEFAULT, dBuffer);

  double neg_one = -1.0;
  hipblasDaxpy(blasHandle, N, &neg_one, b, 1, r0, 1);

  double num = 0.0;
  hipblasDnrm2(blasHandle, N, r0, 1, &num);

  double den = 0.0;
  hipblasDnrm2(blasHandle, N, b, 1, &den);

  hipFree(dBuffer);
  hipsparseDestroySpMat(matA);
  hipsparseDestroyDnVec(vecX);
  hipsparseDestroyDnVec(vecY);
  hipsparseDestroy(spHandle);
  hipblasDestroy(blasHandle);

  return num / den;
}
#endif

#ifndef __USPARSE_BUILD_BACKEND_ROCM__
double Verification(SpM<double> *d_A, double *x, double *b, double *r0, int N) {
  return 0.0;
}
#endif
