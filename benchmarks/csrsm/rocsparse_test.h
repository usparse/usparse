#pragma once
#include "common.h"

template <typename T>
T *test_rocsparse_matrix(const MatrixData<T> &matrix_in, const T *h_b, int k, int h_rhs, int *out_m, float *out_time);
