#ifndef SPMV_ACC_VECTOR_ROW_SPMV_COO_H
#define SPMV_ACC_VECTOR_ROW_SPMV_COO_H

// void line_enhance_sparse_spmv_coo(int trans, const double alpha, const double beta,
//                               const int* d_row_index, const int* d_col_index, const double* d_values, 
//                               const int rows, const int cols, const int nnz, 
//                               const double *d_x, double *d_y);
void vector_row_sparse_spmv_coo(int trans, double alpha, double beta,
                                int* d_row_index, int* d_col_index, double* d_values, 
                                int rows, int cols,int nnz, 
                                double *d_x, double *d_y);

#endif