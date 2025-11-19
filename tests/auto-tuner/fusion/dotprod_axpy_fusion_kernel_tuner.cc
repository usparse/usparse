//
// Created by genshen to 2025-08-30.
//

#include <iostream>
#include <random>

#include <usparse_tuner.h>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <rocsparse/rocsparse.h>

#include "../../unit/tests_utils.hpp"
#include "kernels/usparse_gmres.h"

USPARSE_TUNER_BENCH(dotProdAxpyFusion_Tuner) {
  template <int THREADS = 0, int ELES_PER_THREAD = 0, int STRIDES = 0>
  static void utuner(usparse::tuner::States & states, std::tuple<int, int> args) {
    auto [N, I] = args; // N: length of data. I:
    using T = float;    // for FP32.

    constexpr int RESTART_TIMES = 20;

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
    float f_density = 0.5; // TODO: just for pass the compiling

    T *d_r0 = nullptr;
    T *d_V = nullptr;
    T *d_H = nullptr;
    TEST_HIP_CHECK(hipMalloc(&d_r0, N * sizeof(T)));
    TEST_HIP_CHECK(hipMalloc(&d_V, N * (RESTART_TIMES + 1) * sizeof(T)));
    TEST_HIP_CHECK(hipMalloc(&d_H, (RESTART_TIMES + 1) * RESTART_TIMES * sizeof(T)));
    TEST_HIP_CHECK(hipMemcpy(d_r0, h_r0.data(), N * sizeof(T), hipMemcpyHostToDevice));
    TEST_HIP_CHECK(hipMemcpy(d_V, h_V.data(), N * (RESTART_TIMES + 1) * sizeof(T), hipMemcpyHostToDevice));
    TEST_HIP_CHECK(
        hipMemcpy(d_H, h_H.data(), (RESTART_TIMES) * (RESTART_TIMES + 1) * sizeof(T), hipMemcpyHostToDevice));

    std::size_t buffer_size = 0;
    T *d_temp_buffer = nullptr;
    auto VN = usparse::get_r0_v_mem_len(N);
    usparse::dotprod_axpy_fusion<uint, T, usparse::GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION_LEVEL_2>(
        nullptr, nullptr, nullptr, d_temp_buffer, (uint)N, (uint)VN, (uint)m, (uint)i, &buffer_size, f_density);
    TEST_HIP_CHECK(hipMalloc(&d_temp_buffer, buffer_size));

    for (auto _ : states) {
      usparse::dotprod_axpy_fusion_internal_imp<uint, T, THREADS, ELES_PER_THREAD, STRIDES,
                                                usparse::GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION_LEVEL_2>(
          d_r0, d_V, d_H, d_temp_buffer, (uint)N, (uint)VN, (uint)m, (uint)i, nullptr);
      hipDeviceSynchronize();
    }

    hipFree(d_r0);
    hipFree(d_V);
    hipFree(d_H);
    hipFree(d_temp_buffer);

    const std::size_t mem_size = N * sizeof(T) * 2;
    states.counters["Bandwidth"] = usparse::tuner::Counter(
        mem_size * states.iterations(), usparse::tuner::Counter::kIsRate, usparse::tuner::Counter::kIs1024);
  };
};

USPARSE_TUNER_DEFINE_TUNE_RANGE(dotProdAxpyFusion_Tuner) {
  using BlockThreads = usparse::tuner::ParamRange<int, 128, 256, 512>;
  using ElesPerThreads = usparse::tuner::ParamRange<int, 2, 4>;
  using StridesParams = usparse::tuner::ParamRange<int, 2, 4>;
  constexpr int ParamGroups = 3;

  using ConfigType = usparse::tuner::Config<int, int>; // N and I.
  auto config = new ConfigType();
  config->setIterations(16);
  config->setArgs(std::tuple{18000, 3})->setArgs(std::tuple{36000, 2})->setArgs(std::tuple{72000, 2});

  USPARSE_TUNER_LAUNCH(dotProdAxpyFusion_Tuner, ConfigType, config, ParamGroups, BlockThreads, ElesPerThreads,
                       StridesParams);
}
