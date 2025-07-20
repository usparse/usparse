#ifndef SPMV_COO2CSR_H
#define SPMV_COO2CSR_H

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>


void coo2csr(const int m, const int nnz, const int* coo_row_ind, int* csr_row_ptr);

#endif // SPMV_ACC_COO2CSR_H
