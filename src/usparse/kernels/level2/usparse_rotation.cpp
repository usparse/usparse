//
// Created by genshen to 2025-07-09.
//

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "usparse_utils.hpp"
#include "usparse_rotation.h"

namespace usparse {
  template <typename INDEX_TYPE, typename DATA_TYPE>
  __host__ __device__ __forceinline__ void applyRotation(DATA_TYPE &dx, DATA_TYPE &dy, const DATA_TYPE cs,
                                                         const DATA_TYPE sn) {
    DATA_TYPE temp = cs * dx + sn * dy;
    dy = (-sn) * dx + cs * dy;
    dx = temp;
  }

  template <typename INDEX_TYPE, typename DATA_TYPE>
  __host__ __device__ __forceinline__ void generateRotation(const DATA_TYPE dx, const DATA_TYPE dy, DATA_TYPE &cs,
                                                            DATA_TYPE &sn) {
    if (dx == DATA_TYPE(0)) {
      cs = DATA_TYPE(0);
      sn = DATA_TYPE(1);
    } else {
      const DATA_TYPE scale = fabs(dx) + fabs(dy); // todo: may double to float
      const DATA_TYPE norm = scale * sqrt(fabs(dx / scale) * fabs(dx / scale) + fabs(dy / scale) * fabs(dy / scale));
      const DATA_TYPE alpha = dx / fabs(dx);
      cs = fabs(dx) / norm;
      sn = alpha * dy / norm;
    }
  }

  template <typename INDEX_TYPE, typename DATA_TYPE>
  __host__ __device__ __forceinline__ void rotation2_kernel_imp(const INDEX_TYPE Am, DATA_TYPE *H, DATA_TYPE *cs,
                                                                DATA_TYPE *sn, DATA_TYPE *s, INDEX_TYPE i) {
    for (INDEX_TYPE k = 0; k < i; k++) {
      applyRotation<INDEX_TYPE, DATA_TYPE>(H[k], H[(k + 1)], cs[k], sn[k]);
    }
    generateRotation<INDEX_TYPE, DATA_TYPE>(H[i], H[(i + 1)], cs[i], sn[i]);
    applyRotation<INDEX_TYPE, DATA_TYPE>(H[i], H[(i + 1)], cs[i], sn[i]);
    applyRotation<INDEX_TYPE, DATA_TYPE>(s[i], s[i + 1], cs[i], sn[i]);
  }

  template <typename INDEX_TYPE, typename DATA_TYPE>
  __global__ void rotation2_kernel(INDEX_TYPE Am, DATA_TYPE *H, DATA_TYPE *cs, DATA_TYPE *sn, DATA_TYPE *s,
                                   INDEX_TYPE i) {
    rotation2_kernel_imp<INDEX_TYPE, DATA_TYPE>(Am, H, cs, sn, s, i);
  }
} // namespace usparse

template <typename INDEX_TYPE, typename DATA_TYPE>
void usparse::host_rotation2(const INDEX_TYPE Am, DATA_TYPE *d_H, DATA_TYPE *d_cs, DATA_TYPE *d_sn, DATA_TYPE *d_s,
                             INDEX_TYPE i) {
  rotation2_kernel_imp<INDEX_TYPE, DATA_TYPE>(Am, d_H, d_cs, d_sn, d_s, i);
}

template <typename INDEX_TYPE, typename DATA_TYPE>
void usparse::device_rotation2(const INDEX_TYPE Am, DATA_TYPE *d_H, DATA_TYPE *d_cs, DATA_TYPE *d_sn, DATA_TYPE *d_s,
                               INDEX_TYPE i) {
  rotation2_kernel<<<1, 1>>>(Am, d_H, d_cs, d_sn, d_s, i);
}

template <typename INDEX_TYPE, typename DATA_TYPE>
void usparse::host_rotation2_and_sync(const INDEX_TYPE Am, DATA_TYPE *d_H, DATA_TYPE *d_cs, DATA_TYPE *d_sn,
                                      DATA_TYPE *d_s, INDEX_TYPE i, sync_handle &handle) {
  host_rotation2(Am, d_H, d_cs, d_sn, d_s, i);
}

template <typename INDEX_TYPE, typename DATA_TYPE>
void usparse::device_rotation2_and_sync(const INDEX_TYPE Am, DATA_TYPE *d_H, DATA_TYPE *d_cs, DATA_TYPE *d_sn,
                                        DATA_TYPE *d_s, INDEX_TYPE i, sync_handle &handle) {
  rotation2_kernel<<<1, 1>>>(Am, d_H, d_cs, d_sn, d_s, i);
  handle.sync();
}

template void usparse::host_rotation2<uint, double>(const uint Am, double *d_H, double *d_cs, double *d_sn, double *d_s,
                                                    uint i);

template void usparse::host_rotation2<uint, float>(const uint Am, float *d_H, float *d_cs, float *d_sn, float *d_s,
                                                   uint i);

template void usparse::host_rotation2_and_sync<uint, double>(const uint Am, double *d_H, double *d_cs, double *d_sn,
                                                             double *d_s, uint i, sync_handle &handle);

template void usparse::host_rotation2_and_sync<uint, float>(const uint Am, float *d_H, float *d_cs, float *d_sn,
                                                            float *d_s, uint i, sync_handle &handle);

template void usparse::device_rotation2<uint, double>(const uint Am, double *d_H, double *d_cs, double *d_sn,
                                                      double *d_s, uint i);

template void usparse::device_rotation2<uint, float>(const uint Am, float *d_H, float *d_cs, float *d_sn, float *d_s,
                                                     uint i);

template void usparse::device_rotation2_and_sync<uint, double>(const uint Am, double *d_H, double *d_cs, double *d_sn,
                                                               double *d_s, uint i, sync_handle &handle);

template void usparse::device_rotation2_and_sync<uint, float>(const uint Am, float *d_H, float *d_cs, float *d_sn,
                                                              float *d_s, uint i, sync_handle &handle);
