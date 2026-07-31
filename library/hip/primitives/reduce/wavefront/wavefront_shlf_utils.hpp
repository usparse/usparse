
//
// Created by genshen to 2025-08-28.
//

#ifndef USPARSE_KERNELS_USPARSE_PRIMITIVES_WAVEFRONT_SHLF_UTILS_HPP
#define USPARSE_KERNELS_USPARSE_PRIMITIVES_WAVEFRONT_SHLF_UTILS_HPP

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "building_config.h"
#include "rocsparse/rocsparse-complex-types.h"

__device__ __forceinline__ usparse::usparse_fpc32 __shfl_down(usparse::usparse_fpc32 var, unsigned int lane_delta, int width) {
  float r = __shfl_down(std::real(var), lane_delta, width);
  float i = __shfl_down(std::imag(var), lane_delta, width);
  return usparse::usparse_fpc32{r, i};
}

__device__ __forceinline__ usparse::usparse_fpc64 __shfl_down(usparse::usparse_fpc64 var, unsigned int lane_delta, int width) {
  double r = __shfl_down(std::real(var), lane_delta, width);
  double i = __shfl_down(std::imag(var), lane_delta, width);
  return usparse::usparse_fpc64{r, i};
}

#ifdef __USPARSE_BUILD_BACKEND_CUDA__
__device__ __forceinline__ usparse::usparse_fpc32 __shfl_down(usparse::usparse_fpc32 var, int lane_delta, int width) {
  return __shfl_down(var, (unsigned int)lane_delta, width);
}

__device__ __forceinline__ usparse::usparse_fpc64 __shfl_down(usparse::usparse_fpc64 var, int lane_delta, int width) {
  return __shfl_down(var, (unsigned int)lane_delta, width);
}
#endif

#endif // USPARSE_KERNELS_USPARSE_PRIMITIVES_WAVEFRONT_SHLF_UTILS_HPP
