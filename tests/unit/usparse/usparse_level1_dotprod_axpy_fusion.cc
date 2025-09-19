//
// Created by genshen to 2025-08-12.
//

#include <cmath>
#include <random>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <rocsparse/rocsparse.h>

#include "tests_utils.hpp"
#include "usparse_gmres.h"

double dotProduct(const double *x, const double *y, uint N) {
  double sum = 0.0;
  for (uint i = 0; i < N; ++i) {
    sum += x[i] * y[i];
  }
  return sum;
}

void daxpy(double alpha, const double *x, double *y, uint N) {
  for (uint i = 0; i < N; ++i) {
    y[i] += alpha * x[i];
  }
}

void dcopy(const double *src, double *dst, uint N) {
  for (uint i = 0; i < N; ++i) {
    dst[i] = src[i];
  }
}

double calculateNorm(const double *vec, uint N) {
  double sum = 0.0;
  for (uint i = 0; i < N; ++i) {
    sum += vec[i] * vec[i];
  }
  return std::sqrt(sum);
}

void dscal(double alpha, double *x, uint N) {
  for (uint i = 0; i < N; ++i) {
    x[i] *= alpha;
  }
}

void print_arr(const double *src, const uint start, const uint end) {
  for (uint i = start; i < end; ++i) {
    printf("Host, i:%d, v:%f\n", i, src[i]);
  }
}

TEST(Level1Test, dotProdAxpyFusion) {
  constexpr int N = 100000;
  constexpr int RESTART_TIMES = 20;
  constexpr int I = 2;

  std::vector<double> h_r0(N);
  std::vector<double> h_V((RESTART_TIMES + 1) * N);
  std::vector<double> h_H((RESTART_TIMES + 1) * (RESTART_TIMES));

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
  const float f_density = 0.5; // TODO: just for passing the compiling.

  for (int k = 0; k <= i; k++) {
    h_H[k * RESTART_TIMES + i] = dotProduct(h_r0.data(), h_V.data() + k * N, N);
    const double alpha = -h_H[k * RESTART_TIMES + i];
    daxpy(alpha, h_V.data() + N * k, h_r0.data(), N);
  }
  h_H[(i + 1) * RESTART_TIMES + i] = calculateNorm(h_r0.data(), N);
  const double alpha = 1.0 / h_H[(i + 1) * RESTART_TIMES + i];
  dscal(alpha, h_r0.data(), N);
  dcopy(h_r0.data(), h_V.data() + N * (i + 1), N);

  std::vector<double> r0_expected(N);
  dcopy(h_r0.data(), r0_expected.data(), N);

  // reset values
  gen.seed(2025);
  for (int j = 0; j < N; j++) {
    h_r0[j] = dis(gen); // or 1 or j + 1;
    for (int k = 0; k < RESTART_TIMES + 1; k++) {
      h_V[k * N + j] = dis(gen); // or 1 or (j + 1) * (k + 1);
    }
  }

  double *d_r0 = nullptr;
  double *d_V = nullptr;
  double *d_H = nullptr;
  // double *d_result = nullptr;
  TEST_HIP_CHECK(hipMalloc(&d_r0, N * sizeof(double)));
  TEST_HIP_CHECK(hipMalloc(&d_V, N * (RESTART_TIMES + 1) * sizeof(double)));
  TEST_HIP_CHECK(hipMalloc(&d_H, (RESTART_TIMES + 1) * RESTART_TIMES * sizeof(double)));
  TEST_HIP_CHECK(hipMemcpy(d_r0, h_r0.data(), N * sizeof(double), hipMemcpyHostToDevice));
  TEST_HIP_CHECK(hipMemcpy(d_V, h_V.data(), N * (RESTART_TIMES + 1) * sizeof(double), hipMemcpyHostToDevice));
  TEST_HIP_CHECK(
      hipMemcpy(d_H, h_H.data(), (RESTART_TIMES) * (RESTART_TIMES + 1) * sizeof(double), hipMemcpyHostToDevice));

  std::size_t buffer_size = 0;
  double *d_temp_buffer = nullptr;
  usparse::dotprod_axpy_fusion<uint, double, usparse::GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION_LEVEL_2>(
      nullptr, nullptr, nullptr, d_temp_buffer, (uint)N, (uint)m, (uint)i, &buffer_size, f_density);
  TEST_HIP_CHECK(hipMalloc(&d_temp_buffer, buffer_size));

  usparse::dotprod_axpy_fusion<uint, double, usparse::GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION_LEVEL_2>(
      d_r0, d_V, d_H, d_temp_buffer, (uint)N, (uint)m, (uint)i, 0, f_density);
  hipDeviceSynchronize();

  // check result.
  double dev_result_on_host = 0; // result: H[0,0]
  TEST_HIP_CHECK(
      hipMemcpy(&dev_result_on_host, d_H + (0 * RESTART_TIMES + 0), 1 * sizeof(double), hipMemcpyDeviceToHost));
  EXPECT_DOUBLE_EQ(dev_result_on_host, h_H[0 * RESTART_TIMES + 0]);

  double dev_h_on_host = 0; //  result: H[1, 0]
  TEST_HIP_CHECK(hipMemcpy(&dev_h_on_host, d_H + (1 * RESTART_TIMES + 0), 1 * sizeof(double), hipMemcpyDeviceToHost));
  EXPECT_DOUBLE_EQ(dev_h_on_host, h_H[1 * RESTART_TIMES + 0]);

  std::vector<double> r0_dev_result(N); // result : r0[N]
  TEST_HIP_CHECK(hipMemcpy(r0_dev_result.data(), d_r0, N * sizeof(double), hipMemcpyDeviceToHost));

  int count = 0;
  printf("----------------------\nNo. host_result dev_result:\n");
  for (int k = 0; k < N; ++k) {
    EXPECT_NEAR(r0_dev_result[k], r0_expected[k], 1e-9);
    if (count < 10) {
      printf("%d %f %f.\n", k, r0_expected[k], r0_dev_result[k]);
    }
    count++;
  }
  printf("----------------------\n");

  std::vector<double> vi_result(N); // result : r0[N]
  TEST_HIP_CHECK(hipMemcpy(vi_result.data(), d_V + (i + 1) * N, N * sizeof(double), hipMemcpyDeviceToHost));

  int cnt = 0;
  printf("----------------------\nNo. v[i+1] dev_result:\n");
  for (int k = 0; k < N; ++k) {
    EXPECT_NEAR(vi_result[k], r0_expected[k], 1e-9);
    if (cnt < 10) {
      printf("%d %f %f.\n", k, r0_expected[k], vi_result[k]);
    }
    cnt++;
  }
  printf("----------------------\n");

  hipFree(d_r0);
  hipFree(d_V);
  hipFree(d_H);
  hipFree(d_temp_buffer);
}
