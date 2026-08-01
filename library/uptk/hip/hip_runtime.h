/**
 * Created by genshen on 2026/4/29.
 */

#ifndef USPARSE_HIP_RUNTIME_H
#define USPARSE_HIP_RUNTIME_H

#include "building_config.h"

#ifdef __USPARSE_BUILD_BACKEND_CUDA__
#include <cuda_runtime_api.h>
#endif

#ifdef __USPARSE_BUILD_BACKEND_MUSA__
#include <musa_runtime_api.h>
#endif

#include <building_config.h>

// in recent cuda version, __shfl_down api is removed.
template <typename T>
__device__ __forceinline__ T __shfl_down(T var, unsigned int lane_delta, const int width = __WF_SIZE__) {
  return __shfl_down_sync(0xffffffff, var, (unsigned int)lane_delta, width);
}

template <typename T>
__device__ __forceinline__ T __shfl_up(T var, unsigned int lane_delta, const int width = __WF_SIZE__) {
  return __shfl_up_sync(0xffffffff, var, (unsigned int)lane_delta, width);
}

template <typename T> __device__ __forceinline__ T __shfl(T var, unsigned int src_lane, int width = __WF_SIZE__) {
  return __shfl_sync(0xffffffff, var, src_lane, width);
}

#define hipThreadIdx_x threadIdx.x
#define hipThreadIdx_y threadIdx.y
#define hipThreadIdx_z threadIdx.z

#define hipBlockDim_x blockDim.x
#define hipBlockDim_y blockDim.y
#define hipBlockDim_z blockDim.z

#define hipBlockIdx_x blockIdx.x
#define hipBlockIdx_y blockIdx.y
#define hipBlockIdx_z blockIdx.z

#define hipGridDim_x gridDim.x
#define hipGridDim_y gridDim.y
#define hipGridDim_z gridDim.z

#endif // USPARSE_HIP_RUNTIME_H
