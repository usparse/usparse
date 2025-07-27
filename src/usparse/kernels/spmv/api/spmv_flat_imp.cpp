//
// Created by yi on 2024/7/18.
//

#include <iostream>

#include "../hip-flat/spmv_hip_acc_imp.h"
#include "spmv_flat.h"
#include "types.h"

void spmv_flat(int trans, const double alpha, const double beta,
               const csr_desc<int, double> h_csr_desc,
               const csr_desc<int, double> d_csr_desc, const double *x,
               double *y) {
  flat_sparse_spmv(trans, alpha, beta, h_csr_desc, d_csr_desc, x, y);
}
