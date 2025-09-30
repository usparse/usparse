//
// Created by genshen to 2025-07-09.
//

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "usparse_solver_tri.h"

namespace usparse {
  template <typename INDEX_TYPE, typename DATA_TYPE>
  __host__ __device__ __forceinline__ void sovlerTri_kernel_imp(const INDEX_TYPE Am, const INDEX_TYPE i,
                                                                const DATA_TYPE *H, DATA_TYPE *s) {
    for (int j = i; j >= 0; j--) {
      s[j] /= H[Am * j + j];
      for (int k = j - 1; k >= 0; k--) {
        s[k] -= H[k * Am + j] * s[j];
      }
    }
  }

  template <typename INDEX_TYPE, typename DATA_TYPE>
  __global__ void sovlerTri_kernel(const INDEX_TYPE Am, const INDEX_TYPE i, const DATA_TYPE *H, DATA_TYPE *s) {
    sovlerTri_kernel_imp(Am, i, H, s);
  }
} // namespace usparse

template <typename INDEX_TYPE, typename DATA_TYPE>
void usparse::host_sovlerTri(const INDEX_TYPE Am, const INDEX_TYPE i, const DATA_TYPE *H, DATA_TYPE *s) {
  usparse::sovlerTri_kernel_imp(Am, i, H, s);
}

template <typename INDEX_TYPE, typename DATA_TYPE>
void usparse::device_sovlerTri(const INDEX_TYPE Am, const INDEX_TYPE i, const DATA_TYPE *H, DATA_TYPE *s) {
  usparse::sovlerTri_kernel<<<1, 1>>>(Am, i, H, s);
}

template void usparse::host_sovlerTri<int, double>(const int Am, int i, const double *H, double *s);

template void usparse::host_sovlerTri<int, float>(const int Am, int i, const float *H, float *s);

template void usparse::device_sovlerTri<int, double>(const int Am, int i, const double *H, double *s);

template void usparse::device_sovlerTri<int, float>(const int Am, const int i, const float *H, float *s);
