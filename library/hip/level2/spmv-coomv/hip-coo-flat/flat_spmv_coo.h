#ifndef SPMV_ACC_FLAT_SPMV_COO_H
#define SPMV_ACC_FLAT_SPMV_COO_H

void flat_sparse_spmv_coo(int trans, double alpha, double beta, 
                          int* d_row_index, int* d_col_index, double* d_values, 
                          int rows, int cols, int nnz, 
                          double *d_x, double *d_y);
#endif