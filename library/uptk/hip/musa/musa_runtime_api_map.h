/**
 * Created by genshen on 2026/07/31.
 */
#ifndef USPARSE_HIP_RUNTIME_API_MUSA_H
#define USPARSE_HIP_RUNTIME_API_MUSA_H

#include <musa_runtime.h>
#include <musa_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Type Aliases
// ============================================================================

// Error handling types
using hipError_t = musaError_t;

// Core object types
using hipStream_t = musaStream_t;
using hipEvent_t = musaEvent_t;

// Memory copy direction enumeration
using hipMemcpyKind = musaMemcpyKind;

// ============================================================================
// Constants & Enums Mapping
// ============================================================================

// Success status
constexpr hipError_t hipSuccess = musaSuccess;

// Memory copy directions
constexpr hipMemcpyKind hipMemcpyHostToHost = musaMemcpyHostToHost;
constexpr hipMemcpyKind hipMemcpyHostToDevice = musaMemcpyHostToDevice;
constexpr hipMemcpyKind hipMemcpyDeviceToHost = musaMemcpyDeviceToHost;
constexpr hipMemcpyKind hipMemcpyDeviceToDevice = musaMemcpyDeviceToDevice;
constexpr hipMemcpyKind hipMemcpyDefault = musaMemcpyDefault;

// Event flags
constexpr unsigned int hipEventDefault = musaEventDefault;
constexpr unsigned int hipEventBlockingSync = musaEventBlockingSync;
constexpr unsigned int hipEventDisableTiming = musaEventDisableTiming;
constexpr unsigned int hipEventInterprocess = musaEventInterprocess;

// ============================================================================
// Error Handling APIs
// ============================================================================

inline const char *hipGetErrorString(hipError_t error) { return musaGetErrorString(error); }

inline const char *hipGetErrorName(hipError_t error) { return musaGetErrorName(error); }

inline hipError_t hipGetLastError() { return musaGetLastError(); }

// ============================================================================
// Device Management APIs
// ============================================================================
using hipDeviceProp_t = musaDeviceProp;

inline hipError_t hipSetDevice(int deviceId) { return musaSetDevice(deviceId); }

inline hipError_t hipGetDevice(int *deviceId) { return musaGetDevice(deviceId); }

inline hipError_t hipDeviceSynchronize() { return musaDeviceSynchronize(); }

inline hipError_t hipGetDeviceCount(int *count) { return musaGetDeviceCount(count); }

inline hipError_t hipGetDeviceProperties(hipDeviceProp_t *prop, int device) {
  return musaGetDeviceProperties(prop, device);
}

// ============================================================================
// Stream Management APIs
// ============================================================================
constexpr hipStream_t hipStreamDefault = musaStreamDefault;

inline hipError_t hipStreamCreate(hipStream_t *stream) { return musaStreamCreate(stream); }

inline hipError_t hipStreamDestroy(hipStream_t stream) { return musaStreamDestroy(stream); }

inline hipError_t hipStreamSynchronize(hipStream_t stream) { return musaStreamSynchronize(stream); }

// ============================================================================
// Memory Management APIs
// ============================================================================
#define hipHostMallocDefault musaHostAllocDefault

inline hipError_t hipMalloc(void **ptr, size_t size) { return musaMalloc(ptr, size); }

inline hipError_t hipHostMalloc(void **ptr, size_t size, unsigned int flags = hipHostMallocDefault) {
  return musaHostAlloc(ptr, size, flags);
}

inline hipError_t hipMallocAsync(void **dev_ptr, size_t size, hipStream_t stream) {
  return musaMallocAsync(dev_ptr, size, stream);
}

inline hipError_t hipFree(void *ptr) { return musaFree(ptr); }

inline hipError_t hipFreeAsync(void *ptr, hipStream_t stream) { return hipFree(ptr); } // todo:

inline hipError_t hipMemcpy(void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind) {
  return musaMemcpy(dst, src, sizeBytes, kind);
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
  return musaMemcpyAsync(dst, src, sizeBytes, kind, stream);
}

inline hipError_t hipMemset(void *dst, int value, size_t sizeBytes) { return musaMemset(dst, value, sizeBytes); }

inline hipError_t hipMemsetAsync(void *dst, int value, size_t sizeBytes, hipStream_t stream = 0) {
  return musaMemsetAsync(dst, value, sizeBytes, stream);
}

// ============================================================================
// Event Management APIs
// ============================================================================

inline hipError_t hipEventCreate(hipEvent_t *event) { return musaEventCreate(event); }

inline hipError_t hipEventCreateWithFlags(hipEvent_t *event, unsigned int flags) {
  return musaEventCreateWithFlags(event, flags);
}

inline hipError_t hipEventDestroy(hipEvent_t event) { return musaEventDestroy(event); }

inline hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream = 0) { return musaEventRecord(event, stream); }

inline hipError_t hipEventSynchronize(hipEvent_t event) { return musaEventSynchronize(event); }

inline hipError_t hipEventElapsedTime(float *ms, hipEvent_t start, hipEvent_t stop) {
  return musaEventElapsedTime(ms, start, stop);
}

// ============================================================================
// Kernel Launch Support - Macro wrapper for <<< >>> syntax
// ============================================================================

// Make hipLaunchKernelGGL equivalent to standard musa kernel <<< >>> invocation
#define hipLaunchKernelGGL(kernelName, numBlocks, numThreads, memPerBlock, streamId, ...)                              \
  do {                                                                                                                 \
    kernelName<<<(numBlocks), (numThreads), (memPerBlock), (streamId)>>>(__VA_ARGS__);                                 \
  } while (0)

#ifdef __cplusplus
}
#endif

#endif // USPARSE_HIP_RUNTIME_API_MUSA_H
