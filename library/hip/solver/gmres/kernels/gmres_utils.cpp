//
// Created by genshen to 2025-09-18.
//

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "hip/common/usparse_utils.hpp"

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
    USPARSE_HIP_API_CHECK(hipMalloc((void **)&d_data, N * sizeof(int)));
    warmup_kernel<<<N / 64, 64>>>(d_data, N);
    USPARSE_HIP_API_CHECK(hipDeviceSynchronize());
    USPARSE_HIP_API_CHECK(hipFree(d_data));
  }
} // namespace utils

