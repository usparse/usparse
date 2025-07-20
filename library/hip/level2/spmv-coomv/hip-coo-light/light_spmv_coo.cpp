#include "light_spmv_coo.h"
#include "../../../conversion/coo2csr.h"
#include "../api/types.h"
#include "../spmv-acc-csrmv/hip-light/spmv_hip_acc_imp.h"

void light_sparse_spmv_coo(int trans, double alpha, double beta, int *d_row_index, int *d_col_index, double *d_values,
                           int rows, int cols, int nnz, double *d_x, double *d_y) {
  int *d_row_ptr;
  hipMalloc((void **)&d_row_ptr, sizeof(int) * (rows + 1));
  var_csr_desc<int, double> d_csr_desc;
  d_csr_desc.rows = rows;
  d_csr_desc.cols = cols;
  d_csr_desc.nnz = nnz;
  d_csr_desc.col_index = d_col_index;
  d_csr_desc.values = d_values;
  coo2csr(rows, nnz, d_row_index, d_row_ptr);
  d_csr_desc.row_ptr = d_row_ptr;
  csr_desc<int, double> csr = d_csr_desc.as_const();
  light_sparse_spmv(trans, alpha, beta, csr, d_x, d_y);
  hipDeviceSynchronize();
  hipFree(d_row_ptr);
}