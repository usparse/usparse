/**
 * Created by genshen on 2026/4/29.
 */

#ifndef USPARSE_HIP_COOPERATIVE_GROUPS_H
#define USPARSE_HIP_COOPERATIVE_GROUPS_H

#include <cooperative_groups.h>

#ifdef hipLaunchCooperativeKernel
#undef hipLaunchCooperativeKernel
#endif

#include "building_config.h"

#ifdef __USPARSE_BUILD_BACKEND_CUDA__
template <typename F>
inline cudaError_t hipLaunchCooperativeKernel(F f, dim3 gridDim, dim3 blockDim, void **kernelParams,
                                              unsigned int sharedMemBytes, cudaStream_t stream) {
  return cudaLaunchCooperativeKernel((const void *)f, gridDim, blockDim, kernelParams, (size_t)sharedMemBytes, stream);
}
#endif

#ifdef __USPARSE_BUILD_BACKEND_MUSA__
template <typename F>
inline musaError_t hipLaunchCooperativeKernel(F f, dim3 gridDim, dim3 blockDim, void **kernelParams,
                                              unsigned int sharedMemBytes, musaStream_t stream) {
  return musaLaunchCooperativeKernel((const void *)f, gridDim, blockDim, kernelParams, (size_t)sharedMemBytes, stream);
}
#endif

#endif // USPARSE_HIP_COOPERATIVE_GROUPS_H
