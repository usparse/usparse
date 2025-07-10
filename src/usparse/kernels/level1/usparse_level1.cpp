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
  __global__ void __launch_bounds__(512)
      daxpy_kernel(const DATA_TYPE alpha, const DATA_TYPE *x, DATA_TYPE *y, const INDEX_TYPE N) {
    const int tid = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (tid < N) {
      y[tid] += alpha * x[tid];
    }
  }

  template <typename INDEX_TYPE, typename DATA_TYPE>
  __global__ void __launch_bounds__(512) dscal_kernel(const DATA_TYPE alpha, DATA_TYPE *x, const INDEX_TYPE N) {
    const int tid = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (tid < N) {
      x[tid] *= alpha;
    }
  }

  template <typename INDEX_TYPE, typename DATA_TYPE>
  __global__ void __launch_bounds__(512) dcopy_kernel(const DATA_TYPE *src, DATA_TYPE *dst, const INDEX_TYPE N) {
    const int tid = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (tid < N) {
      dst[tid] = src[tid];
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
void usparse::device_daxpy(const DATA_TYPE alpha, const DATA_TYPE *x, DATA_TYPE *y, const INDEX_TYPE N) {
  constexpr int THREADS_PER_BLOCK = 512;
  const int BLOCKS = N / THREADS_PER_BLOCK + (N % THREADS_PER_BLOCK == 0 ? 0 : 1);
  daxpy_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(alpha, x, y, N);
}

template <typename INDEX_TYPE, typename DATA_TYPE>
void usparse::device_dscal(const DATA_TYPE alpha, DATA_TYPE *x, const INDEX_TYPE N) {
  constexpr int THREADS_PER_BLOCK = 512;
  const int BLOCKS = N / THREADS_PER_BLOCK + (N % THREADS_PER_BLOCK == 0 ? 0 : 1);
  dscal_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(alpha, x, N);
}

template <typename INDEX_TYPE, typename DATA_TYPE>
void usparse::device_dcopy(const DATA_TYPE *src, DATA_TYPE *dst, const INDEX_TYPE N) {
  constexpr int THREADS_PER_BLOCK = 512;
  const int BLOCKS = N / THREADS_PER_BLOCK + (N % THREADS_PER_BLOCK == 0 ? 0 : 1);
  dcopy_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(src, dst, N);
}

template void usparse::device_calculateNorm<uint, double>(const double *d_vec, uint N, double *d_result);
template void usparse::device_calculateNorm<uint, float>(const float *d_vec, uint N, float *d_result);

template void usparse::device_dotProduct<uint, double>(const double *d_x, const double *d_y, uint N, double *d_result);
template void usparse::device_dotProduct<uint, float>(const float *d_x, const float *d_y, uint N, float *d_result);

template void usparse::device_daxpy<uint, double>(const double alpha, const double *x, double *y, const uint N);
template void usparse::device_daxpy<uint, float>(const float alpha, const float *x, float *y, const uint N);

template void usparse::device_dscal<uint, double>(const double alpha, double *x, const uint N);
template void usparse::device_dscal<uint, float>(const float alpha, float *x, const uint N);

template void usparse::device_dcopy<uint, double>(const double *src, double *dst, const uint N);
template void usparse::device_dcopy<uint, float>(const float *src, float *dst, const uint N);
