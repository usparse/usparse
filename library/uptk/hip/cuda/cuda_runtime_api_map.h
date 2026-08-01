/**
 * Created by genshen on 2026/04/28.
 */
#ifndef USPARSE_HIP_RUNTIME_API_CUDA_MAP_H
#define USPARSE_HIP_RUNTIME_API_CUDA_MAP_H

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Type Aliases
// ============================================================================

// Error handling types
using hipError_t = cudaError_t;

// Core object types
using hipStream_t = cudaStream_t;
using hipEvent_t = cudaEvent_t;

// Memory copy direction enumeration
using hipMemcpyKind = cudaMemcpyKind;

// ============================================================================
// Constants & Enums Mapping
// ============================================================================

// Success status
constexpr hipError_t hipSuccess = cudaSuccess;

// Memory copy directions
constexpr hipMemcpyKind hipMemcpyHostToHost = cudaMemcpyHostToHost;
constexpr hipMemcpyKind hipMemcpyHostToDevice = cudaMemcpyHostToDevice;
constexpr hipMemcpyKind hipMemcpyDeviceToHost = cudaMemcpyDeviceToHost;
constexpr hipMemcpyKind hipMemcpyDeviceToDevice = cudaMemcpyDeviceToDevice;
constexpr hipMemcpyKind hipMemcpyDefault = cudaMemcpyDefault;

// Event flags
constexpr unsigned int hipEventDefault = cudaEventDefault;
constexpr unsigned int hipEventBlockingSync = cudaEventBlockingSync;
constexpr unsigned int hipEventDisableTiming = cudaEventDisableTiming;
constexpr unsigned int hipEventInterprocess = cudaEventInterprocess;

// ============================================================================
// Error Handling APIs
// ============================================================================

inline const char *hipGetErrorString(hipError_t error) { return cudaGetErrorString(error); }

inline const char *hipGetErrorName(hipError_t error) { return cudaGetErrorName(error); }

inline hipError_t hipGetLastError() { return cudaGetLastError(); }

// ============================================================================
// Device Management APIs
// ============================================================================
using hipDeviceProp_t = cudaDeviceProp;

inline hipError_t hipSetDevice(int deviceId) { return cudaSetDevice(deviceId); }

inline hipError_t hipGetDevice(int *deviceId) { return cudaGetDevice(deviceId); }

inline hipError_t hipDeviceSynchronize() { return cudaDeviceSynchronize(); }

inline hipError_t hipGetDeviceCount(int *count) { return cudaGetDeviceCount(count); }

inline hipError_t hipGetDeviceProperties(hipDeviceProp_t *prop, int device) {
  return cudaGetDeviceProperties(prop, device);
}

// ============================================================================
// Stream Management APIs
// ============================================================================
constexpr hipStream_t hipStreamDefault = cudaStreamDefault;

inline hipError_t hipStreamCreate(hipStream_t *stream) { return cudaStreamCreate(stream); }

inline hipError_t hipStreamDestroy(hipStream_t stream) { return cudaStreamDestroy(stream); }

inline hipError_t hipStreamSynchronize(hipStream_t stream) { return cudaStreamSynchronize(stream); }

// ============================================================================
// Memory Management APIs
// ============================================================================
#define hipHostMallocDefault cudaHostAllocDefault

inline hipError_t hipMalloc(void **ptr, size_t size) { return cudaMalloc(ptr, size); }

inline hipError_t hipHostMalloc(void **ptr, size_t size, unsigned int flags = hipHostMallocDefault) {
  return cudaHostAlloc(ptr, size, flags);
}

inline hipError_t hipMallocAsync(void **dev_ptr, size_t size, hipStream_t stream) {
  return cudaMallocAsync(dev_ptr, size, stream);
}

inline hipError_t hipFree(void *ptr) { return cudaFree(ptr); }

inline hipError_t hipFreeAsync(void *ptr, hipStream_t stream) { return cudaFreeAsync(ptr, stream); }

inline hipError_t hipMemcpy(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind) {
  return cudaMemcpy(dst, src, sizeBytes, kind);
}

inline hipError_t hipMemcpyHtoD(void *dst, const void *src, size_t sizeBytes) {
  return hipMemcpy(dst, src, sizeBytes, hipMemcpyHostToDevice);
}

inline hipError_t hipMemcpyDtoH(void *dst, const void *src, size_t sizeBytes) {
  return hipMemcpy(dst, src, sizeBytes, hipMemcpyDeviceToHost);
}

inline hipError_t hipMemcpyDtoD(void *dst, const void *src, size_t sizeBytes) {
  return hipMemcpy(dst, src, sizeBytes, hipMemcpyDeviceToDevice);
}

inline hipError_t hipMemcpyHtoH(void *dst, const void *src, size_t sizeBytes) {
  return hipMemcpy(dst, src, sizeBytes, hipMemcpyHostToHost);
}

inline hipError_t hipMemcpyAsync(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind,
                                 hipStream_t stream = 0) {
  return cudaMemcpyAsync(dst, src, sizeBytes, kind, stream);
}

inline hipError_t hipMemset(void *dst, int value, size_t sizeBytes) { return cudaMemset(dst, value, sizeBytes); }

inline hipError_t hipMemsetAsync(void *dst, int value, size_t sizeBytes, hipStream_t stream = 0) {
  return cudaMemsetAsync(dst, value, sizeBytes, stream);
}

// ============================================================================
// Event Management APIs
// ============================================================================

inline hipError_t hipEventCreate(hipEvent_t *event) { return cudaEventCreate(event); }

inline hipError_t hipEventCreateWithFlags(hipEvent_t *event, unsigned int flags) {
  return cudaEventCreateWithFlags(event, flags);
}

inline hipError_t hipEventDestroy(hipEvent_t event) { return cudaEventDestroy(event); }

inline hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream = 0) { return cudaEventRecord(event, stream); }

inline hipError_t hipEventSynchronize(hipEvent_t event) { return cudaEventSynchronize(event); }

inline hipError_t hipEventElapsedTime(float *ms, hipEvent_t start, hipEvent_t stop) {
  return cudaEventElapsedTime(ms, start, stop);
}

// ============================================================================
// Kernel Launch Support - Macro wrapper for <<< >>> syntax
// ============================================================================

// Make hipLaunchKernelGGL equivalent to standard CUDA kernel <<< >>> invocation
#define hipLaunchKernelGGL(kernelName, numBlocks, numThreads, memPerBlock, streamId, ...)                              \
  do {                                                                                                                 \
    kernelName<<<(numBlocks), (numThreads), (memPerBlock), (streamId)>>>(__VA_ARGS__);                                 \
  } while (0)

#ifdef __cplusplus
}
#endif

#endif // USPARSE_HIP_RUNTIME_API_CUDA_MAP_H
