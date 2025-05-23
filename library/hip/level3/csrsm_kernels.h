//
// Created by limingjun on 2025/5/17.
//

#ifndef USPARSE_HIP_LEVEL3_CSRSM_H
#define USPARSE_HIP_LEVEL3_CSRSM_H


#include <vector>
#include <hip/hip_runtime.h>

constexpr int WARP_SIZE = 64;

template <typename T> struct MatrixData {
  std::vector<int> row_ptr;
  std::vector<int> col_index;
  std::vector<T> values;
  int rows;
  int cols;
  int nnz;
  bool valid = true;
  std::string error_msg;
};

template <typename T>
__global__ void sptrsm_updown_kernel(const int *row_ptr, const int *col_idx, const T *val, const T *b, T *x,
                                     int *is_solved, int n, int rhs, bool is_lower);

template <typename T>
T *test_updown_matrix(const MatrixData<T> &matrix_in, const T *h_b, int k, int h_rhs, int *n_out, bool is_lower,
                      float *out_time);

#endif // USPARSE_HIP_LEVEL3_CSRSM_H
