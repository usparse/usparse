//
// Created by genshen to 2025-07-13.
//

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <rocsparse/rocsparse.h>

#include "tests_utils.hpp"
#include "usparse_gmres.h"

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
  TEST_HIP_CHECK(hipMalloc(&d_x, N * sizeof(double)));
  TEST_HIP_CHECK(hipMalloc(&d_y, N * sizeof(double)));
  TEST_HIP_CHECK(hipMalloc(&d_result, 1 * sizeof(double)));
  TEST_HIP_CHECK(hipMemset(d_result, 0, 1 * sizeof(double)));
  TEST_HIP_CHECK(hipMemcpy(d_x, h_x, N * sizeof(double), hipMemcpyHostToDevice));
  TEST_HIP_CHECK(hipMemcpy(d_y, h_y, N * sizeof(double), hipMemcpyHostToDevice));

  usparse::device_dotProduct<uint, double>(d_x, d_y, (uint)N, d_result);
  hipDeviceSynchronize();

  double dev_result_on_host = 0;
  TEST_HIP_CHECK(hipMemcpy(&dev_result_on_host, d_result, 1 * sizeof(double), hipMemcpyDeviceToHost));

  // check result.
  EXPECT_DOUBLE_EQ(dev_result_on_host, h_dot_pro);

  hipFree(d_x);
  hipFree(d_y);
  hipFree(d_result);

  delete[] h_x;
  delete[] h_y;
}
