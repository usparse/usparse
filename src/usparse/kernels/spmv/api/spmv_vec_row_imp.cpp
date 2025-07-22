//
// Created by yi on 2024/7/18.
//

#include <iostream>

#include "../hip-vector-row/vector_row.h"
#include "spmv_flat.h"
#include "types.h"

void spmv_vector_row(int trans, const double alpha, const double beta,
                     const csr_desc<int, double> d_csr_desc, const double *x,
                     double *y) {
  vec_row_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
}
