//
// Created by genshen on 2021/07/06.
//

#ifndef SPMV_ACC_SPMV_HIP_ACC_IMP_FLAT_H
#define SPMV_ACC_SPMV_HIP_ACC_IMP_FLAT_H

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "flat_imp_one_pass.hpp"
#include "../api/types.h"

struct csrmv_flat_mtx_info {
  int *break_points = nullptr;
};

template <int WF_SIZE, int R, int BLOCKS, int THREADS, typename I, typename T>
__global__ void spmv_flat_kernel(int m, const T alpha, const T beta, const I *__restrict__ row_offset,
                                 const I *__restrict__ break_points, const I *__restrict__ csr_col_ind,
                                 const T *__restrict__ csr_val, const T *__restrict__ x, T *__restrict__ y);

template <int BREAK_STRIDE, int BLOCKS, typename I>
__global__ void pre_calc_break_point(const I *__restrict__ row_ptr, const I m, I *__restrict__ break_points,
                                     const I bp_len);

template <typename FPType, int BEAT_Y_POLICY = BEAT_Y_POLICY_AUTO>
void adaptive_flat_sparse_spmv(int trans, const FPType alpha, const FPType beta, const csr_desc<int, FPType> d_csr_desc,
                               const FPType *x, FPType *y, const csrmv_flat_mtx_info *info);

template <typename FPType>
void adaptive_flat_analyse(const csr_desc<int, FPType> d_csr_desc, csrmv_flat_mtx_info *info);

#include "flat_imp.inl"

#endif // SPMV_ACC_SPMV_HIP_ACC_IMP_FLAT_H
