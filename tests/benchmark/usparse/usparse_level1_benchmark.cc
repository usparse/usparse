//
// Created by genshen to 2025-07-14.
//

#include <iostream>

#include <benchmark/benchmark.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#ifdef __HIP_PLATFORM_HCC__
#include <rocsparse/rocsparse.h>
#endif

#include "kernels/usparse_gmres.h"
#include "level1/usparse_blas_level1.h"
#include "usparse_level1_fixture.hpp"

// benchmarking usparse::device_calculateNorm
BENCHMARK_TEMPLATE_METHOD_F(UsparseLevel1Fixture, CalcNorm)(benchmark::State &state) {
  using I = typename Base::IndexType;
  using T = typename Base::DataType;

  const int N = state.range(0);

  // prepare data
  T *d_x = Base::dev_x;
  T *d_result = nullptr;
  TEST_HIP_CHECK(hipMalloc((void **)&d_result, 1 * sizeof(T)));
  TEST_HIP_CHECK(hipMemset(d_result, 0, 1 * sizeof(T)));

  std::size_t buffer_size = 0;
  double *d_temp_buffer = nullptr;
  usparse::device_calculateNorm<uint, double>(nullptr, (uint)N, nullptr, d_temp_buffer, &buffer_size);
  TEST_HIP_CHECK(hipMalloc((void **)&d_temp_buffer, (N / 256 + (N % 256 == 0 ? 0 : 1)) * sizeof(double)));

  // benchmark
  for (auto _ : state) {
    usparse::device_calculateNorm<I, T>(d_x, (I)N, d_result, d_temp_buffer, 0);
    TEST_HIP_CHECK(hipDeviceSynchronize());
  }

  // clean
  TEST_HIP_CHECK(hipFree(d_result));
  TEST_HIP_CHECK(hipFree(d_temp_buffer));

  // calc bandwidth
  const std::size_t mem_size = N * sizeof(T);
  state.counters["Bandwidth"] =
      benchmark::Counter(mem_size * state.iterations(), benchmark::Counter::kIsRate, benchmark::Counter::kIs1024);
}

BENCHMARK_TEMPLATE_INSTANTIATE_F(UsparseLevel1Fixture, CalcNorm, uint, double)
    ->Arg(100000)
    ->Arg(1000000)
    ->Arg(10000000)
    ->Arg(100000000)
    ->UseRealTime();

// benchmarking usparse::device_dotProduct
BENCHMARK_TEMPLATE_METHOD_F(UsparseLevel1Fixture, dotprod)(benchmark::State &state) {
  using I = typename Base::IndexType;
  using T = typename Base::DataType;

  const int N = state.range(0);

  // prepare data
  // the device vector of size N
  T *d_x = Base::dev_x;
  T *d_y = Base::dev_y;
  // prepare another data
  T *d_result = nullptr;
  TEST_HIP_CHECK(hipMalloc((void **)&d_result, 1 * sizeof(T)));
  TEST_HIP_CHECK(hipMemset(d_result, 0, 1 * sizeof(T)));

  // parepare data: temp buffer for reduction.
  std::size_t buffer_size = 0;
  double *d_temp_buffer = nullptr;
  usparse::device_dotProduct<uint, double>(nullptr, nullptr, (uint)N, nullptr, d_temp_buffer, &buffer_size);
  TEST_HIP_CHECK(hipMalloc((void **)&d_temp_buffer, buffer_size));

  // benchmark
  for (auto _ : state) {
    usparse::device_dotProduct<I, T>(d_x, d_y, (I)N, d_result, d_temp_buffer, 0);
    TEST_HIP_CHECK(hipDeviceSynchronize());
  }

  // clean
  TEST_HIP_CHECK(hipFree(d_result));
  TEST_HIP_CHECK(hipFree(d_temp_buffer));

  // calc bandwidth
  const std::size_t mem_size = N * sizeof(T) * 2;
  state.counters["Bandwidth"] =
      benchmark::Counter(mem_size * state.iterations(), benchmark::Counter::kIsRate, benchmark::Counter::kIs1024);
}

BENCHMARK_TEMPLATE_INSTANTIATE_F(UsparseLevel1Fixture, dotprod, uint, double)
    ->Arg(100000)
    ->Arg(1000000)
    ->Arg(10000000)
    ->Arg(100000000)
    ->UseRealTime();

// benchmarking usparse::device_daxpy
BENCHMARK_TEMPLATE_METHOD_F(UsparseLevel1Fixture, daxpy)(benchmark::State &state) {
  using I = typename Base::IndexType;
  using T = typename Base::DataType;

  const int N = state.range(0);

  // prepare data:
  const T alpha = 0.5;
  // the device vectors of size N
  T *d_x = Base::dev_x;
  T *d_y = Base::dev_y;

  // benchmark
  for (auto _ : state) {
    // y = y + alpha * x
    usparse::device_daxpy<I, T>(alpha, d_x, d_y, (I)N);
    TEST_HIP_CHECK(hipDeviceSynchronize());
  }

  // calc bandwidth
  const std::size_t mem_size = N * sizeof(T) * 3;
  state.counters["Bandwidth"] =
      benchmark::Counter(mem_size * state.iterations(), benchmark::Counter::kIsRate, benchmark::Counter::kIs1024);
}

BENCHMARK_TEMPLATE_INSTANTIATE_F(UsparseLevel1Fixture, daxpy, uint, double)
    ->Arg(100000)
    ->Arg(1000000)
    ->Arg(10000000)
    ->Arg(100000000)
    ->UseRealTime();

// benchmarking usparse::device_dscal
BENCHMARK_TEMPLATE_METHOD_F(UsparseLevel1Fixture, dscal)(benchmark::State &state) {
  using I = typename Base::IndexType;
  using T = typename Base::DataType;

  const int N = state.range(0);

  // prepare data
  const T alpha = 0.5;
  // the device vector of size N
  T *d_x = Base::dev_x;

  // benchmark
  for (auto _ : state) {
    // x = alpha * x
    usparse::device_dscal<I, T>(alpha, d_x, (I)N);
    TEST_HIP_CHECK(hipDeviceSynchronize());
  }

  // calc bandwidth
  const std::size_t mem_size = N * sizeof(T) * 2;
  state.counters["Bandwidth"] =
      benchmark::Counter(mem_size * state.iterations(), benchmark::Counter::kIsRate, benchmark::Counter::kIs1024);
}

BENCHMARK_TEMPLATE_INSTANTIATE_F(UsparseLevel1Fixture, dscal, uint, double)
    ->Arg(100000)
    ->Arg(1000000)
    ->Arg(10000000)
    ->Arg(100000000)
    ->UseRealTime();

// benchmarking usparse::device_dcopy
BENCHMARK_TEMPLATE_METHOD_F(UsparseLevel1Fixture, dcopy)(benchmark::State &state) {
  using I = typename Base::IndexType;
  using T = typename Base::DataType;

  const int N = state.range(0);

  T *d_src = Base::dev_x;
  T *d_dest = Base::dev_y;

  // benchmark
  for (auto _ : state) {
    usparse::device_dcopy<I, T>(d_src, d_dest, (I)N);
    TEST_HIP_CHECK(hipDeviceSynchronize());
  }

  // calc bandwidth
  const std::size_t mem_size = N * sizeof(T) * 2;
  state.counters["Bandwidth"] =
      benchmark::Counter(mem_size * state.iterations(), benchmark::Counter::kIsRate, benchmark::Counter::kIs1024);
}

BENCHMARK_TEMPLATE_INSTANTIATE_F(UsparseLevel1Fixture, dcopy, uint, double)
    ->Arg(100000)
    ->Arg(1000000)
    ->Arg(10000000)
    ->Arg(100000000)
    ->UseRealTime();

BENCHMARK_MAIN();
