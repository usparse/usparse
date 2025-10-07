//
// Created by genshen to 2025-08-13.
//

#include <iostream>
#include <random>

#include <benchmark/benchmark.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <rocsparse/rocsparse.h>

#include "../../unit/tests_utils.hpp"
#include "usparse_gmres.h"

// benchmarking usparse::dotprod_axpy_fusion
template <typename T> static void Usparse_dotprod_axpy_fusion(benchmark::State &state) {
  const int N = state.range(0);
  constexpr int RESTART_TIMES = 20;
  constexpr int I = 2;

  std::vector<T> h_r0(N);
  std::vector<T> h_V((RESTART_TIMES + 1) * N);
  std::vector<T> h_H((RESTART_TIMES + 1) * (RESTART_TIMES));

  // initial values
  std::mt19937 gen(2025); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis(-1.0, 1.0);
  for (int j = 0; j < N; j++) {
    h_r0[j] = dis(gen); // or 1 or j + 1;
    for (int k = 0; k < RESTART_TIMES + 1; k++) {
      h_V[k * N + j] = dis(gen); // or 1 or (j + 1) * (k + 1);
    }
  }

  for (int j = 0; j < RESTART_TIMES + 1; j++) {
    for (int k = 0; k < RESTART_TIMES; k++) {
      h_H[k * RESTART_TIMES + j] = 0.0;
    }
  }

  const int i = I; // todo: change this value (i must <= RESTART_TIMES)
  const int m = RESTART_TIMES;

  const float f_density = 0.5; // TODO: just for passing the compiling

  T *d_r0 = nullptr;
  T *d_V = nullptr;
  T *d_H = nullptr;
  TEST_HIP_CHECK(hipMalloc(&d_r0, N * sizeof(T)));
  TEST_HIP_CHECK(hipMalloc(&d_V, N * (RESTART_TIMES + 1) * sizeof(T)));
  TEST_HIP_CHECK(hipMalloc(&d_H, (RESTART_TIMES + 1) * RESTART_TIMES * sizeof(T)));
  TEST_HIP_CHECK(hipMemcpy(d_r0, h_r0.data(), N * sizeof(T), hipMemcpyHostToDevice));
  TEST_HIP_CHECK(hipMemcpy(d_V, h_V.data(), N * (RESTART_TIMES + 1) * sizeof(T), hipMemcpyHostToDevice));
  TEST_HIP_CHECK(hipMemcpy(d_H, h_H.data(), (RESTART_TIMES) * (RESTART_TIMES + 1) * sizeof(T), hipMemcpyHostToDevice));

  std::size_t buffer_size = 0;
  T *d_temp_buffer = nullptr;
  auto VN = usparse::get_r0_v_mem_len(N);
  usparse::dotprod_axpy_fusion<uint, T, usparse::GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION_LEVEL_2>(
      nullptr, nullptr, nullptr, d_temp_buffer, (uint)N, (uint)VN, (uint)m, (uint)i, &buffer_size, f_density);
  TEST_HIP_CHECK(hipMalloc(&d_temp_buffer, buffer_size));

  for (auto _ : state) {
    usparse::dotprod_axpy_fusion<uint, T, usparse::GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION_LEVEL_2>(
        d_r0, d_V, d_H, d_temp_buffer, (uint)N, (uint)VN, (uint)m, (uint)i, 0, f_density);
    hipDeviceSynchronize();
  }

  hipFree(d_r0);
  hipFree(d_V);
  hipFree(d_H);
  hipFree(d_temp_buffer);

  const std::size_t mem_size = N * sizeof(T) * 2;
  state.counters["Bandwidth"] =
      benchmark::Counter(mem_size * state.iterations(), benchmark::Counter::kIsRate, benchmark::Counter::kIs1024);
}

BENCHMARK_TEMPLATE(Usparse_dotprod_axpy_fusion, float)
    ->Arg(18000)
    ->Arg(36000)
    ->Arg(72000)
    ->Arg(160000)
    ->Arg(320000)
    ->Arg(640000)
    ->Arg(1280000)
    ->Arg(3200000)
    ->Arg(3600000)
    ->UseRealTime();

template <typename T> static void Usparse_dotprod_axpy_normal(benchmark::State &state) {
  constexpr int N = 100000;
  constexpr int RESTART_TIMES = 20;
  constexpr int I = 2;

  std::vector<T> h_r0(N);
  std::vector<T> h_V((RESTART_TIMES + 1) * N);
  std::vector<T> h_H((RESTART_TIMES + 1) * (RESTART_TIMES));

  // initial values
  std::mt19937 gen(2025); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis(-1.0, 1.0);
  for (int j = 0; j < N; j++) {
    h_r0[j] = dis(gen); // or 1 or j + 1;
    for (int k = 0; k < RESTART_TIMES + 1; k++) {
      h_V[k * N + j] = dis(gen); // or 1 or (j + 1) * (k + 1);
    }
  }

  for (int j = 0; j < RESTART_TIMES + 1; j++) {
    for (int k = 0; k < RESTART_TIMES; k++) {
      h_H[k * RESTART_TIMES + j] = 0.0;
    }
  }

  const int i = I; // todo: change this value (i must <= RESTART_TIMES)
  const int m = RESTART_TIMES;

  T *d_r0 = nullptr;
  T *d_V = nullptr;
  T *d_H = nullptr;
  TEST_HIP_CHECK(hipMalloc(&d_r0, N * sizeof(T)));
  TEST_HIP_CHECK(hipMalloc(&d_V, N * (RESTART_TIMES + 1) * sizeof(T)));
  TEST_HIP_CHECK(hipMalloc(&d_H, (RESTART_TIMES + 1) * RESTART_TIMES * sizeof(T)));
  TEST_HIP_CHECK(hipMemcpy(d_r0, h_r0.data(), N * sizeof(T), hipMemcpyHostToDevice));
  TEST_HIP_CHECK(hipMemcpy(d_V, h_V.data(), N * (RESTART_TIMES + 1) * sizeof(T), hipMemcpyHostToDevice));
  TEST_HIP_CHECK(hipMemcpy(d_H, h_H.data(), (RESTART_TIMES) * (RESTART_TIMES + 1) * sizeof(T), hipMemcpyHostToDevice));

  std::size_t buffer_size = N; // too large N.
  T *d_temp_buffer = nullptr;
  TEST_HIP_CHECK(hipMalloc(&d_temp_buffer, buffer_size));

  for (auto _ : state) {
    for (int k = 0; k <= i; k++) {
      usparse::device_dotProduct<uint, T>(d_r0, d_V + k * N, N, &d_H[k * RESTART_TIMES + i], d_temp_buffer, 0);
      hipDeviceSynchronize();
      const T alpha = -d_H[k * RESTART_TIMES + i];
      usparse::device_daxpy<uint, T, T>(alpha, d_V + N * k, d_r0, N);
    }
    hipDeviceSynchronize();
  }

  hipFree(d_r0);
  hipFree(d_V);
  hipFree(d_H);
  hipFree(d_temp_buffer);
}

BENCHMARK_TEMPLATE(Usparse_dotprod_axpy_normal, float)
    ->Arg(18000)
    ->Arg(36000)
    ->Arg(72000)
    ->Arg(160000)
    ->Arg(320000)
    ->Arg(640000)
    ->Arg(1280000)
    ->Arg(3200000)
    ->Arg(3600000)
    ->UseRealTime();

// BENCHMARK_TEMPLATE(Usparse_dotprod_axpy_fusion, double);

/*
BENCHMARK_TEMPLATE_METHOD_F(UsparseLevel1Fixture, dotDaxpyFusion)(benchmark::State &state) {
  using I = typename Base::IndexType;
  using T = typename Base::DataType;

  const uint N = state.range(0);
  const uint m = 1;
  const uint i = 0;

  // prepare data
  // the device vector of size N
  T *d_x = Base::dev_x;
  T *d_y = Base::dev_y;
  // prepare another data
  T *d_result = nullptr;
  TEST_HIP_CHECK(hipMalloc(&d_result, N * sizeof(T)));
  TEST_HIP_CHECK(hipMemset(d_result, 0, N * sizeof(T)));

  // parepare data: temp buffer for reduction.
  std::size_t buffer_size = 0;
  double *d_temp_buffer = nullptr;
  usparse::dot_daxpy_fusion<uint, double>(nullptr, nullptr, nullptr, d_temp_buffer, N, m, i, &buffer_size);
  TEST_HIP_CHECK(hipMalloc(&d_temp_buffer, buffer_size));

  // benchmark
  for (auto _ : state) {
    usparse::dot_daxpy_fusion<I, T>(d_x, d_y, d_result, d_temp_buffer, N, m, i, &buffer_size);
    hipDeviceSynchronize();
  }

  // clean
  TEST_HIP_CHECK(hipFree(d_result));
  TEST_HIP_CHECK(hipFree(d_temp_buffer));

  // calc bandwidth
  const std::size_t mem_size = N * sizeof(T) * 2;
  state.counters["Bandwidth"] =
      benchmark::Counter(mem_size * state.iterations(), benchmark::Counter::kIsRate, benchmark::Counter::kIs1024);
}

BENCHMARK_TEMPLATE_INSTANTIATE_F(UsparseLevel1Fixture, dotDaxpyFusion, uint, double)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000)
    ->Arg(1000000)
    ->UseRealTime();
*/
