//
// Created by genshen on 2021/7/15.
//

#include "api/spmv.h"
#include "building_config.h"

#include "adaptive/adaptive.h"
#include "flat/csrmv_flat.h"
#include "light/spmv_hip_acc_imp.h"
#include "line-enhance/csrmv_line_enhance.h"
#include "line/line_strategy.h"
#include "thread-row/thread_row.h"
#include "vector-row/vector_row.h"

void sparse_csr_spmv(int trans, const double alpha, const double beta, const csr_desc<int, double> h_csr_desc,
                     const csr_desc<int, double> d_csr_desc, const double *x, double *y) {
#ifdef KERNEL_STRATEGY_DEFAULT
  adaptive_sparse_spmv(trans, alpha, beta, h_csr_desc, d_csr_desc, x, y);
#endif

#ifdef KERNEL_STRATEGY_ADAPTIVE
  adaptive_sparse_spmv(trans, alpha, beta, h_csr_desc, d_csr_desc, x, y);
#endif


#ifdef KERNEL_STRATEGY_FLAT
#ifndef FLAT_SEGMENT_SUM_REDUCE
  flat_sparse_spmv(trans, alpha, beta, h_csr_desc, d_csr_desc, x, y);
#endif
#ifdef FLAT_SEGMENT_SUM_REDUCE
  segment_sum_flat_sparse_spmv(trans, alpha, beta, h_csr_desc, d_csr_desc, x, y);
#endif
#endif

#ifdef KERNEL_STRATEGY_LIGHT
  light_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
#endif

#ifdef KERNEL_STRATEGY_LINE
  adaptive_line_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
#endif

#ifdef KERNEL_STRATEGY_THREAD_ROW
  thread_row_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
#endif

#ifdef KERNEL_STRATEGY_VECTOR_ROW
  vec_row_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
#endif

#ifdef KERNEL_STRATEGY_LINE_ENHANCE
  line_enhance_sparse_spmv(trans, alpha, beta, d_csr_desc, x, y);
#endif
}
