//
// Created by genshen to 2025-07-09.
//

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "usparse_utils.hpp"

#ifndef USPARSE_KERNELS_LEVEL2_USPARSE_ROTATION_H
#define USPARSE_KERNELS_LEVEL2_USPARSE_ROTATION_H

namespace usparse {
  template <typename INDEX_TYPE, typename DATA_TYPE>
  void host_rotation2(const INDEX_TYPE Am, DATA_TYPE *d_H, DATA_TYPE *d_cs, DATA_TYPE *d_sn, DATA_TYPE *d_s,
                      INDEX_TYPE i);

  template <typename INDEX_TYPE, typename DATA_TYPE>
  void device_rotation2(const INDEX_TYPE Am, DATA_TYPE *d_H, DATA_TYPE *d_cs, DATA_TYPE *d_sn, DATA_TYPE *d_s,
                        INDEX_TYPE i);

  template <typename INDEX_TYPE, typename DATA_TYPE>
  void host_rotation2_and_sync(const INDEX_TYPE Am, DATA_TYPE *d_H, DATA_TYPE *d_cs, DATA_TYPE *d_sn, DATA_TYPE *d_s,
                               INDEX_TYPE i, sync_handle &handle);

  template <typename INDEX_TYPE, typename DATA_TYPE>
  void device_rotation2_and_sync(const INDEX_TYPE Am, DATA_TYPE *d_H, DATA_TYPE *d_cs, DATA_TYPE *d_sn, DATA_TYPE *d_s,
                                 INDEX_TYPE i, sync_handle &handle, INDEX_TYPE N, INDEX_TYPE NNZ);
} // namespace usparse

#endif // USPARSE_KERNELS_LEVEL2_USPARSE_ROTATION_H
