//
// Created by genshen to 2025-09-18.
//

#ifndef USPARSE_GMRES_UTILS_HPP
#define USPARSE_GMRES_UTILS_HPP

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

__global__ void warmup_kernel(int *data, const int N) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
    data[idx] = 1;
  }
}

namespace utils {
  void warmup() {
    int *d_data = nullptr;
    const int N = 4096 * 64;
    hipMalloc(&d_data, N * sizeof(int));
    warmup_kernel<<<N / 64, 64>>>(d_data, N);
    hipDeviceSynchronize();
    hipFree(d_data);
  }
} // namespace utils

#endif // USPARSE_GMRES_UTILS_HPP
