//
// Created by yi on 2024/7/18.
//

#include <iostream>

#include "../hip-line-enhance/line_enhance_spmv.h"
#include "spmv_flat.h"
#include "types.h"

void spmv_line_enhance(int trans, const double alpha, const double beta,
                       const csr_desc<int, double> d_csr_desc, const double *x,
                       double *y) {
  adaptive_enhance_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
}
