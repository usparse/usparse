#include "adaptive_spmv_coo.h"
#include "../../../conversion/coo2csr.h"
#include "../api/types.h"
#include "../spmv-acc-csrmv/hip-adaptive/adaptive.h"

void adaptive_sparse_spmv_coo(int trans, double alpha, double beta, int *d_row_index, int *d_col_index,
                              double *d_values, int rows, int cols, int nnz, double *d_x, double *d_y) {
  int *d_row_ptr;
  hipMalloc((void **)&d_row_ptr, sizeof(int) * (rows + 1));
  var_csr_desc<int, double> d_csr_desc;
  d_csr_desc.rows = rows;
  d_csr_desc.cols = cols;
  d_csr_desc.nnz = nnz;
  d_csr_desc.col_index = d_col_index;
  d_csr_desc.values = d_values;
  var_csr_desc<int, double> h_csr_desc;
  h_csr_desc.rows = rows;
  h_csr_desc.cols = cols;
  h_csr_desc.nnz = nnz;
  h_csr_desc.row_ptr = new int[rows + 1];
  coo2csr(rows, nnz, d_row_index, d_row_ptr);
  d_csr_desc.row_ptr = d_row_ptr;
  hipDeviceSynchronize();
  hipMemcpy(h_csr_desc.row_ptr, d_row_ptr, sizeof(int) * (rows + 1), hipMemcpyDeviceToHost);
  adaptive_sparse_spmv(trans, alpha, beta, h_csr_desc, d_csr_desc, d_x, d_y);
  hipDeviceSynchronize();
  hipFree(d_row_ptr);
  delete[] h_csr_desc.row_ptr;
}
