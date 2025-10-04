//
// Created by genshen on 2025/10/04.
//

#ifndef USPARSE_USPARSE_UTILS_HPP
#define USPARSE_USPARSE_UTILS_HPP

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <iostream>

#define HIP_CHECK(stat)                                                                                                \
  {                                                                                                                    \
    hipError_t err = stat;                                                                                             \
    if (err != hipSuccess) {                                                                                           \
      std::cerr << "Error: HIP reports " << hipGetErrorString(err) << std::endl;                                       \
      std::cerr << "Raised in file " << __FILE__ << "#" << __func__ << "#line " << __LINE__ << std::endl;              \
      std::abort();                                                                                                    \
    }                                                                                                                  \
  }

struct sync_handle {
  static constexpr int SYNC_POLICY = 2;

  static constexpr int SYNC_POLICY_DEVICE_SYNC = 0;
  static constexpr int SYNC_POLICY_EVENT_SYNC = 1;
  static constexpr int SYNC_POLICY_MEMCOPY_SYNC = 2;

  inline sync_handle() {
    if (SYNC_POLICY == SYNC_POLICY_DEVICE_SYNC) {
      return;
    } else if (SYNC_POLICY == SYNC_POLICY_EVENT_SYNC) {
      hipEventCreate(&sync_kernel_event);
    } else if (SYNC_POLICY == SYNC_POLICY_MEMCOPY_SYNC) {
      mem_malloc();
    }
  }

  inline void sync() {
    if (SYNC_POLICY == SYNC_POLICY_DEVICE_SYNC) {
      hipDeviceSynchronize();
    } else if (SYNC_POLICY == SYNC_POLICY_EVENT_SYNC) {
      hipEventRecord(sync_kernel_event);
      hipEventSynchronize(sync_kernel_event);
    } else if (SYNC_POLICY == SYNC_POLICY_MEMCOPY_SYNC) {
      mem_copy();
    }
  }

private:
  int *host_data;
  int *dev_data;

  hipEvent_t sync_kernel_event;

private:
  void mem_malloc() {
    HIP_CHECK(hipHostMalloc((int **)&host_data, 1 * sizeof(int)));
    HIP_CHECK(hipMalloc((int **)&dev_data, 1 * sizeof(int)));
  }

  void mem_copy() {
    // HIP_CHECK(hipMemcpy(dev_data, host_data, 1 * sizeof(int), hipMemcpyHostToDevice));
    hipMemcpyHtoD(dev_data, host_data, 1 * sizeof(int)); // TODO: HIP_CHECK
  }
};

#endif // USPARSE_USPARSE_UTILS_HPP
