//
// Created by genshen to 2025-08-28.
//

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "usparse_blas_level1.h"

namespace usparse {

  template <typename INDEX_TYPE, typename DATA_TYPE>
  __global__ void __launch_bounds__(512) dscal_kernel(const DATA_TYPE alpha, DATA_TYPE *x, const INDEX_TYPE N) {
    const int tid = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (tid < N) {
      x[tid] *= alpha;
    }
  }

  template <typename INDEX_TYPE, typename DATA_TYPE, int THREADS>
  __global__ void __launch_bounds__(512) sacle_set_kernel(const DATA_TYPE alpha, DATA_TYPE data[], const INDEX_TYPE N) {
    const int tid = THREADS * hipBlockIdx_x + hipThreadIdx_x;
    if (tid < N) {
      data[tid] = alpha;
    }
  }
} // namespace usparse

template <typename INDEX_TYPE, typename DATA_TYPE>
void usparse::device_dscal(const DATA_TYPE alpha, DATA_TYPE *x, const INDEX_TYPE N) {
  constexpr int THREADS_PER_BLOCK = 512;
  const int BLOCKS = N / THREADS_PER_BLOCK + (N % THREADS_PER_BLOCK == 0 ? 0 : 1);
  dscal_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(alpha, x, N);
}

template <typename INDEX_TYPE, typename DATA_TYPE>
void usparse::device_scale_set(const DATA_TYPE alpha, DATA_TYPE *data, const INDEX_TYPE N) {
  constexpr unsigned int THREADS = 256;
  const unsigned int grid_size = (N + THREADS - 1) / THREADS;

  (sacle_set_kernel<INDEX_TYPE, DATA_TYPE, THREADS>)<<<grid_size, THREADS>>>(alpha, data, N);
}

template void usparse::device_scale_set<uint, double>(const double alpha, double *data, const uint N);
template void usparse::device_scale_set<uint, float>(const float alpha, float *data, const uint N);
template void usparse::device_scale_set<int, double>(const double alpha, double *data, const int N);
template void usparse::device_scale_set<int, float>(const float alpha, float *data, const int N);

template void usparse::device_dscal<uint, double>(const double alpha, double *x, const uint N);
template void usparse::device_dscal<uint, float>(const float alpha, float *x, const uint N);
template void usparse::device_dscal<int, double>(const double alpha, double *x, const int N);
template void usparse::device_dscal<int, float>(const float alpha, float *x, const int N);
