//
// Created by genshen to 2025-07-15.
//

#ifndef BENCHMARK_TEST_USPARSE_LEVEL1_FIXTURE_HPP
#define BENCHMARK_TEST_USPARSE_LEVEL1_FIXTURE_HPP

#include <iostream>

#include <benchmark/benchmark.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "../../unit/tests_utils.hpp"

/**
 * In this class, two vector of size N is created on both host side and device side.
 * I is index type and T is data type
 * The fixup class for benchmarking Blas level1.
 */
template <typename I, typename T> class UsparseLevel1Fixture : public benchmark::Fixture {
public:
  using IndexType = I;
  using DataType = T;

  void SetUp(::benchmark::State &state) {
    const int N = state.range(0);
    malloc_vector_host_device(h_x, dev_x, N);
    malloc_vector_host_device(h_y, dev_y, N);
  }

  void TearDown(::benchmark::State &state) {
    free_vector_host_device(h_x, dev_x);
    free_vector_host_device(h_y, dev_y);
  }

private:
  // prepare data: malloc host memory space and init them.
  // malloc host and device vector of size N.
  void malloc_vector_host_device(T *(&host_vec), T *(&dev_vec), const std::size_t N) {
    host_vec = new T[N];
    for (int i = 0; i < N; i++) {
      host_vec[i] = i;
    }

    TEST_HIP_CHECK(hipMalloc(&dev_vec, N * sizeof(T)));
    TEST_HIP_CHECK(hipMemcpy(dev_vec, host_vec, N * sizeof(T), hipMemcpyHostToDevice));
  }

  // release host and device memory space
  void free_vector_host_device(T *host_vec, T *dev_vec) {
    hipFree(dev_vec);
    delete[] host_vec;
  }

protected:
  // the vector may be used in benchmark.
  T *h_x = nullptr;
  T *h_y = nullptr;
  T *dev_x = nullptr;
  T *dev_y = nullptr;
};

#endif // BENCHMARK_TEST_USPARSE_LEVEL1_FIXTURE_HPP
