//
// Created by genshen to 2025-07-09.
//

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "usparse_solver_tri.h"

namespace usparse {
  template <typename INDEX_TYPE, typename DATA_TYPE>
  __global__ void sovlerTri_kernel(INDEX_TYPE Am, INDEX_TYPE i, DATA_TYPE *H, DATA_TYPE *s) {
    for (int j = i; j >= 0; j--) {
      s[j] /= H[Am * j + j];
      for (int k = j - 1; k >= 0; k--) {
        s[k] -= H[k * Am + j] * s[j];
      }
    }
  }
} // namespace usparse

template <typename INDEX_TYPE, typename DATA_TYPE>
void usparse::device_sovlerTri(INDEX_TYPE Am, INDEX_TYPE i, DATA_TYPE *H, DATA_TYPE *s) {
  usparse::sovlerTri_kernel<<<1, 1>>>(Am, i, H, s);
}

template void usparse::device_sovlerTri<int, double>(int Am, int i, double *H, double *s);

template void usparse::device_sovlerTri<int, float>(int Am, int i, float *H, float *s);
