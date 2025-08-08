//
// Created by yi on 2024/7/18.
//

#include <iostream>

#include "../hip-line-enhance/line_enhance_spmv.h"
#include "types.h"

template <typename T>
void spmv_line_enhance(int trans, const T alpha, const T beta, const csr_desc<int, T> d_csr_desc, const T *dx, T *dy) {
  adaptive_enhance_sparse_spmv<T>(trans, alpha, beta, d_csr_desc, dx, dy);
}

template void spmv_line_enhance<double>(int, double, double, const csr_desc<int, double>, const double *, double *);
template void spmv_line_enhance<float>(int, float, float, const csr_desc<int, float>, const float *, float *);