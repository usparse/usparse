//
// Created by genshen on 2021/07/06.
//

#ifndef USPARSE_HIP_USPARSE_CSRMV_FLAT_H
#define USPARSE_HIP_USPARSE_CSRMV_FLAT_H

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "usparse_matrix.h"
#include "flat_mtx_info.h"
#include "../../spmv_api_config.hpp"

void flat_sparse_spmv(int trans, const double alpha, const double beta, const csr_desc<int, double> h_csr_desc,
                      const csr_desc<int, double> d_csr_desc, const double *x, double *y,
                      const csrmv_flat_mtx_info *info);

void segment_sum_flat_sparse_spmv(int trans, const double alpha, const double beta, const csr_desc<int, double> h_csr_desc,
                                  const csr_desc<int, double> d_csr_desc, const double *x, double *y);

template <typename FPType, int BETA_Y_POLICY = BETA_Y_POLICY_AUTO>
void adaptive_flat_sparse_spmv(int trans, const FPType alpha, const FPType beta, const csr_desc<int, FPType> d_csr_desc,
                               const FPType *x, FPType *y, const csrmv_flat_mtx_info *info);

template <typename FPType> void flat_analyse(const csr_desc<int, FPType> d_csr_desc, csrmv_flat_mtx_info *info);

#endif // USPARSE_HIP_USPARSE_CSRMV_FLAT_H
