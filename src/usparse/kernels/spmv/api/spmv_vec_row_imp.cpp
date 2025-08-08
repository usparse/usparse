//
// Created by yi on 2024/7/18.
//

#include <iostream>

#include "../hip-vector-row/vector_row.h"
#include "types.h"

template <typename T>
void spmv_vector_row(int trans, const T alpha, const T beta, const csr_desc<int, T> d_csr_desc, const T *dx, T *dy) {
  vec_row_sparse_spmv<T>(trans, alpha, beta, d_csr_desc, dx, dy);
}

template void spmv_vector_row<double>(int, double, double, const csr_desc<int, double>, const double *, double *);
template void spmv_vector_row<float>(int, float, float, const csr_desc<int, float>, const float *, float *);