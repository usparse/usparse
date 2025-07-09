//
// Created by genshen to 2025-07-09.
//

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "usparse_level1.h"

namespace usparse {

  template <typename INDEX_TYPE, typename DATA_TYPE>
  __global__ void calculateNorm_kernel(const DATA_TYPE *vec, INDEX_TYPE N, DATA_TYPE *result) {
    DATA_TYPE sum = 0.0;
    for (INDEX_TYPE i = 0; i < N; ++i) {
      sum += vec[i] * vec[i];
    }
    *result = sqrt(sum);
  }

  template <typename INDEX_TYPE, typename DATA_TYPE>
  __global__ void dotProduct_kernel(const DATA_TYPE *x, const DATA_TYPE *y, INDEX_TYPE N, DATA_TYPE *result) {
    DATA_TYPE sum = 0.0;
    for (INDEX_TYPE i = 0; i < N; ++i) {
      sum += x[i] * y[i];
    }
    *result = sum;
  }

  template <typename INDEX_TYPE, typename DATA_TYPE>
  __global__ void daxpy_kernel(DATA_TYPE alpha, const DATA_TYPE *x, DATA_TYPE *y, INDEX_TYPE N) {
    for (INDEX_TYPE i = 0; i < N; ++i) {
      y[i] += alpha * x[i];
    }
  }

  template <typename INDEX_TYPE, typename DATA_TYPE>
  __global__ void dscal_kernel(DATA_TYPE alpha, DATA_TYPE *x, INDEX_TYPE N) {
    for (INDEX_TYPE i = 0; i < N; ++i) {
      x[i] *= alpha;
    }
  }

  template <typename INDEX_TYPE, typename DATA_TYPE>
  __global__ void dcopy_kernel(const DATA_TYPE *src, DATA_TYPE *dst, INDEX_TYPE N) {
    for (INDEX_TYPE i = 0; i < N; ++i) {
      dst[i] = src[i];
    }
  }

} // namespace usparse

template <typename INDEX_TYPE, typename DATA_TYPE>
void usparse::device_calculateNorm(const DATA_TYPE *d_vec, INDEX_TYPE N, DATA_TYPE *d_result) {
  calculateNorm_kernel<<<1, 1>>>(d_vec, N, d_result);
}

template <typename INDEX_TYPE, typename DATA_TYPE>
void usparse::device_dotProduct(const DATA_TYPE *d_x, const DATA_TYPE *d_y, INDEX_TYPE N, DATA_TYPE *d_result) {
  dotProduct_kernel<<<1, 1>>>(d_x, d_y, N, d_result);
}

template <typename INDEX_TYPE, typename DATA_TYPE>
void usparse::device_daxpy(DATA_TYPE alpha, const DATA_TYPE *x, DATA_TYPE *y, INDEX_TYPE N) {
  daxpy_kernel<<<1, 1>>>(alpha, x, y, N);
}

template <typename INDEX_TYPE, typename DATA_TYPE>
void usparse::device_dscal(DATA_TYPE alpha, DATA_TYPE *x, INDEX_TYPE N) {
  dscal_kernel<<<1, 1>>>(alpha, x, N);
}

template <typename INDEX_TYPE, typename DATA_TYPE>
void usparse::device_dcopy(const DATA_TYPE *src, DATA_TYPE *dst, INDEX_TYPE N) {
  dcopy_kernel<<<1, 1>>>(src, dst, N);
}

template void usparse::device_calculateNorm<uint, double>(const double *d_vec, uint N, double *d_result);
template void usparse::device_calculateNorm<uint, float>(const float *d_vec, uint N, float *d_result);

template void usparse::device_dotProduct<uint, double>(const double *d_x, const double *d_y, uint N, double *d_result);
template void usparse::device_dotProduct<uint, float>(const float *d_x, const float *d_y, uint N, float *d_result);

template void usparse::device_daxpy<uint, double>(double alpha, const double *x, double *y, uint N);
template void usparse::device_daxpy<uint, float>(float alpha, const float *x, float *y, uint N);

template void usparse::device_dscal<uint, double>(double alpha, double *x, uint N);
template void usparse::device_dscal<uint, float>(float alpha, float *x, uint N);

template void usparse::device_dcopy<uint, double>(const double *src, double *dst, uint N);
template void usparse::device_dcopy<uint, float>(const float *src, float *dst, uint N);
