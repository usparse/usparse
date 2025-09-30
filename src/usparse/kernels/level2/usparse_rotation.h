//
// Created by genshen to 2025-07-09.
//

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#ifndef USPARSE_KERNELS_LEVEL2_USPARSE_ROTATION_H
#define USPARSE_KERNELS_LEVEL2_USPARSE_ROTATION_H

namespace usparse {
  template <typename INDEX_TYPE, typename DATA_TYPE>
  void host_rotation2(INDEX_TYPE Am, DATA_TYPE *d_H, DATA_TYPE *d_cs, DATA_TYPE *d_sn, DATA_TYPE *d_s, INDEX_TYPE i);

  template <typename INDEX_TYPE, typename DATA_TYPE>
  void device_rotation2(INDEX_TYPE Am, DATA_TYPE *d_H, DATA_TYPE *d_cs, DATA_TYPE *d_sn, DATA_TYPE *d_s, INDEX_TYPE i);
}

#endif // USPARSE_KERNELS_LEVEL2_USPARSE_ROTATION_H
