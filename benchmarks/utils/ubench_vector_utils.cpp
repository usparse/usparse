//
// Created by genshen on 2025/10/09.
//

#include <iostream>
#include <random>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "ubench_utils.h"
#include "utils.hpp"

namespace ubench {
  template <typename I, typename T> void generate_vector(T *x, int N) {
    std::mt19937 gen;
    gen.seed(1024); // fixed seed
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (int i = 0; i < N; i++) {
      x[i] = static_cast<T>(dis(gen));
    }
  }

  /**
   * malloc host and device vector.
   * copy vector data from host to device.
   */
  template <typename I, typename T, bool MALLOC_HOST>
  void ubench_malloc_and_init_host_dev_vector(T **host_vec, T **dev_vec, I N) {
    if (MALLOC_HOST) {
      (*host_vec) = new T[N];
    }
    generate_vector<I, T>((*host_vec), N); // fill host vector with random number.

    // malloc device data and copy data from host to device.
    HIP_CHECK(hipMalloc((void **)dev_vec, N * sizeof(T)))
    HIP_CHECK(hipMemcpy(*dev_vec, *host_vec, N * sizeof(T), hipMemcpyHostToDevice))
  }

  template <typename I, typename T, bool FREE_HOST> void ubench_free_host_dev_vector(T *host_vec, T *dev_vec, I N) {
    if (FREE_HOST) {
      delete[] host_vec;
    }
    HIP_CHECK(hipFree((void *)dev_vec))
  }

  template void ubench_malloc_and_init_host_dev_vector<usparse::usparse_int, usparse::usparse_fp64>(
      usparse::usparse_fp64 **host_vec, usparse::usparse_fp64 **dev_vec, usparse::usparse_int N);

  template void ubench_malloc_and_init_host_dev_vector<usparse::usparse_int, usparse::usparse_fp32>(
      usparse::usparse_fp32 **host_vec, usparse::usparse_fp32 **dev_vec, usparse::usparse_int N);

} // namespace ubench
