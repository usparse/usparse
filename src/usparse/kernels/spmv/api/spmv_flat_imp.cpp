//
// Created by yi on 2024/7/18.
//

#include <iostream>

#include "../hip-flat/spmv_hip_acc_imp.h"
#include "spmv_flat.h"
#include "types.h"

template <typename T>
void spmv_flat(int trans, const T alpha, const T beta, const csr_desc<int, T> d_csr_desc, const T *dx, T *dy,
               const csrmv_flat_mtx_info *info) {
  // todo: call spmv_analysis first.
  // @deprecated api.
  adaptive_flat_sparse_spmv<T>(trans, alpha, beta, d_csr_desc, dx, dy, info);
}

template void spmv_flat<double>(int, double, double, const csr_desc<int, double>, const double *, double *,
                                const csrmv_flat_mtx_info *);
template void spmv_flat<float>(int, float, float, const csr_desc<int, float>, const float *, float *,
                               const csrmv_flat_mtx_info *);

template <typename T> void flat_analyse(const csr_desc<int, T> d_csr_desc, int **break_points) {
  flat_matrix_analyse<T>(d_csr_desc, break_points);
}
