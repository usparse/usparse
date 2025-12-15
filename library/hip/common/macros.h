//
// Created by genshen on 2021/11/12.
//

#ifndef SPMV_ACC_COMMON_MACROS_H
#define SPMV_ACC_COMMON_MACROS_H

#include "usparse_matrix.h"

#define VAR_FROM_CSR_DESC(T, csr_desc)                                                                                 \
  const int m = csr_desc.rows;                                                                                         \
  const int *rowptr = csr_desc.row_ptr;                                                                                \
  const int *colindex = csr_desc.col_index;                                                                            \
  const T *value = csr_desc.values;

#define MIXED_VAR_FROM_CSR_DESC(I, J, A, csr_desc)                                                                                 \
  const J m = csr_desc.rows;                                                                                         \
  const I *rowptr = csr_desc.row_ptr;                                                                                \
  const J *colindex = csr_desc.col_index;                                                                            \
  const A *value = csr_desc.values;
#endif // SPMV_ACC_COMMON_MACROS_H
