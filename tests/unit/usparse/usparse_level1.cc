//
// Created by genshen to 2025-07-13.
//

#include <cmath>
#include <random>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#ifdef __USPARSE_BUILD_BACKEND_ROCM__
#include <rocsparse/rocsparse.h>
#endif

#include "tests_utils.hpp"
#include "kernels/usparse_gmres.h"
#include "level1/usparse_blas_level1.h"

TEST(Level1Test, dotProduction) {

  constexpr int N = 1000000;
  double *h_x = nullptr, *h_y = nullptr;
  double h_dot_pro = 0;
  h_x = new double[N];
  h_y = new double[N];

  for (int i = 0; i < N; i++) {
    h_x[i] = i;
    h_y[i] = i + 1;
    h_dot_pro += h_x[i] * h_y[i];
  }

  double *d_x = nullptr;
  double *d_y = nullptr;
  double *d_result = nullptr;
  TEST_HIP_CHECK(hipMalloc((void **)&d_x, N * sizeof(double)));
  TEST_HIP_CHECK(hipMalloc((void **)&d_y, N * sizeof(double)));
  TEST_HIP_CHECK(hipMalloc((void **)&d_result, 1 * sizeof(double)));
  TEST_HIP_CHECK(hipMemset(d_result, 0, 1 * sizeof(double)));
  TEST_HIP_CHECK(hipMemcpy(d_x, h_x, N * sizeof(double), hipMemcpyHostToDevice));
  TEST_HIP_CHECK(hipMemcpy(d_y, h_y, N * sizeof(double), hipMemcpyHostToDevice));

  std::size_t buffer_size = 0;
  double *d_temp_buffer = nullptr;
  usparse::device_dotProduct<uint, double>(nullptr, nullptr, (uint)N, nullptr, d_temp_buffer, &buffer_size);
  TEST_HIP_CHECK(hipMalloc((void **)&d_temp_buffer, buffer_size));

  usparse::device_dotProduct<uint, double>(d_x, d_y, (uint)N, d_result, d_temp_buffer, 0);
  TEST_HIP_CHECK(hipDeviceSynchronize());

  double dev_result_on_host = 0;
  TEST_HIP_CHECK(hipMemcpy(&dev_result_on_host, d_result, 1 * sizeof(double), hipMemcpyDeviceToHost));

  // check result.
  EXPECT_DOUBLE_EQ(dev_result_on_host, h_dot_pro);

  TEST_HIP_CHECK(hipFree(d_x));
  TEST_HIP_CHECK(hipFree(d_y));
  TEST_HIP_CHECK(hipFree(d_result));
  TEST_HIP_CHECK(hipFree(d_temp_buffer));

  delete[] h_x;
  delete[] h_y;
}

TEST(Level1Test, vecNorm) {
  constexpr int N = 12000000;
  double *h_vec = nullptr;
  double h_vec_norm = 0;
  h_vec = new double[N];

  std::random_device rd;
  std::default_random_engine eng(rd());
  std::uniform_real_distribution<double> distr(-1.0, 1.0);

  for (int i = 0; i < N; i++) {
    h_vec[i] = distr(eng);
    h_vec_norm += h_vec[i] * h_vec[i];
  }
  h_vec_norm = sqrt(h_vec_norm);

  double *d_vec = nullptr;
  double *d_result = nullptr;
  TEST_HIP_CHECK(hipMalloc((void **)&d_vec, N * sizeof(double)));
  TEST_HIP_CHECK(hipMalloc((void **)&d_result, 1 * sizeof(double)));
  TEST_HIP_CHECK(hipMemset(d_result, 0, 1 * sizeof(double)));
  TEST_HIP_CHECK(hipMemcpy(d_vec, h_vec, N * sizeof(double), hipMemcpyHostToDevice));

  std::size_t buffer_size = 0;
  double *d_temp_buffer = nullptr;
  usparse::device_calculateNorm<uint, double>(nullptr, (uint)N, nullptr, d_temp_buffer, &buffer_size);
  TEST_HIP_CHECK(hipMalloc((void **)&d_temp_buffer, (N / 256 + (N % 256 == 0 ? 0 : 1)) * sizeof(double)));

  usparse::device_calculateNorm<uint, double>(d_vec, (uint)N, d_result, d_temp_buffer, 0);
  TEST_HIP_CHECK(hipDeviceSynchronize());

  double dev_result_on_host = 0;
  TEST_HIP_CHECK(hipMemcpy(&dev_result_on_host, d_result, 1 * sizeof(double), hipMemcpyDeviceToHost));

  // check result.
  // the default EAPECT_DOUBLE_EQ may not pass.
  double relative_error = std::abs(dev_result_on_host - h_vec_norm) / dev_result_on_host;
  EXPECT_NEAR(relative_error, 0.0, 1e-12);

  TEST_HIP_CHECK(hipFree(d_vec));
  TEST_HIP_CHECK(hipFree(d_result));
  TEST_HIP_CHECK(hipFree(d_temp_buffer));

  delete[] h_vec;
}
