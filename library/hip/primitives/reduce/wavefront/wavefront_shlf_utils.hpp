
//
// Created by genshen to 2025-08-28.
//

#ifndef USPARSE_KERNELS_USPARSE_PRIMITIVES_WAVEFRONT_SHLF_UTILS_HPP
#define USPARSE_KERNELS_USPARSE_PRIMITIVES_WAVEFRONT_SHLF_UTILS_HPP

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

__device__ __forceinline__ usparse::usparse_fpc32 __shfl_down(usparse::usparse_fpc32 var, unsigned int lane_delta,
                                                              int width) {
  float r = __shfl_down(std::real(var), lane_delta, width);
  float i = __shfl_down(std::imag(var), lane_delta, width);
  return usparse::usparse_fpc32(r, i);
}

__device__ __forceinline__ usparse::usparse_fpc64 __shfl_down(usparse::usparse_fpc64 var, unsigned int lane_delta,
                                                              int width) {
  double r = __shfl_down(std::real(var), lane_delta, width);
  double i = __shfl_down(std::imag(var), lane_delta, width);
  return usparse::usparse_fpc64(r, i);
}

__device__ __forceinline__ usparse::usparse_fpc32 __shfl_up(usparse::usparse_fpc32 val, unsigned int lane_delta,
                                                            int width) {
  float r = __shfl_up(std::real(val), lane_delta, width);
  float i = __shfl_up(std::imag(val), lane_delta, width);
  return usparse::usparse_fpc32(r, i);
}

__device__ __forceinline__ usparse::usparse_fpc64 __shfl_up(usparse::usparse_fpc64 val, unsigned int lane_delta,
                                                            int width) {
  double r = __shfl_up(std::real(val), lane_delta, width);
  double i = __shfl_up(std::imag(val), lane_delta, width);
  return usparse::usparse_fpc64(r, i);
}

#endif // USPARSE_KERNELS_USPARSE_PRIMITIVES_WAVEFRONT_SHLF_UTILS_HPP
