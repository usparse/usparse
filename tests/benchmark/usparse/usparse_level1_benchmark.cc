//
// Created by genshen to 2025-07-14.
//

#include <iostream>

#include <benchmark/benchmark.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <rocsparse/rocsparse.h>

#include "usparse_gmres.h"
#include "../../unit/tests_utils.hpp"

static void BM_Level1DotProd(benchmark::State &state) {
  const int N = state.range(0);

  // prepare data
  double *h_x = nullptr, *h_y = nullptr;
  h_x = new double[N];
  h_y = new double[N];

  for (int i = 0; i < N; i++) {
    h_x[i] = i;
    h_y[i] = i + 1;
  }

  double *d_x = nullptr;
  double *d_y = nullptr;
  double *d_result = nullptr;
  TEST_HIP_CHECK(hipMalloc(&d_x, N * sizeof(double)));
  TEST_HIP_CHECK(hipMalloc(&d_y, N * sizeof(double)));
  TEST_HIP_CHECK(hipMalloc(&d_result, 1 * sizeof(double)));
  TEST_HIP_CHECK(hipMemset(d_result, 0, 1 * sizeof(double)));
  TEST_HIP_CHECK(hipMemcpy(d_x, h_x, N * sizeof(double), hipMemcpyHostToDevice));
  TEST_HIP_CHECK(hipMemcpy(d_y, h_y, N * sizeof(double), hipMemcpyHostToDevice));

  // benchmark
  for (auto _ : state) {
    usparse::device_dotProduct<uint, double>(d_x, d_y, (uint)N, d_result);
    hipDeviceSynchronize();
  }

  // clean
  hipFree(d_x);
  hipFree(d_y);
  hipFree(d_result);

  delete[] h_x;
  delete[] h_y;

  // calc bandwidth
  const int mem_size = N * sizeof(double) * 2;
  state.counters["Bandwidth"] = benchmark::Counter(mem_size, benchmark::Counter::kIsRate, benchmark::Counter::kIs1024);
}

BENCHMARK(BM_Level1DotProd)->Arg(100000)->Arg(1000000)->Arg(10000000);

BENCHMARK_MAIN();
