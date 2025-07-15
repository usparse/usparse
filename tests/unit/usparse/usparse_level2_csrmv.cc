//
// Created by renyi to 2025-07-16.
//

#include <cmath>
#include <random>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <rocsparse/rocsparse.h>

#include "tests_utils.hpp"
#include "usparse_gmres.h"


TEST(ustbSparseTest, spmv_vector_row) {
  constexpr int N = 100000;
  constexpr int nnz_per_row = 20;
  constexpr int nnz = N * nnz_per_row;

  double *h_x = new double[N];
  double *h_y = new double[N];
  double *h_y_ref = new double[N];
  int *h_row_ptr = new int[N + 1];
  int *h_col_index = new int[nnz];
  double *h_values = new double[nnz];

  for (int i = 0; i < N; i++) {
    h_row_ptr[i] = i * nnz_per_row;
    h_x[i] = 1.0 + (i % 7);

    for (int j = 0; j < nnz_per_row; j++) {
      int col = i - 2 + j;
      if (col < 0)
        col += N;
      if (col >= N)
        col -= N;
      h_col_index[i * nnz_per_row + j] = col;
      h_values[i * nnz_per_row + j] = (i == col) ? 4.0 : 1.0;
    }
  }
  h_row_ptr[N] = nnz;

  std::fill(h_y_ref, h_y_ref + N, 0.0);
  for (int i = 0; i < N; i++) {
    for (int j = h_row_ptr[i]; j < h_row_ptr[i + 1]; j++) {
      h_y_ref[i] += h_values[j] * h_x[h_col_index[j]];
    }
  }

  int *d_row_ptr = nullptr;
  int *d_col_index = nullptr;
  double *d_values = nullptr;
  double *d_x = nullptr;
  double *d_y = nullptr;

  TEST_HIP_CHECK(hipMalloc(&d_row_ptr, (N + 1) * sizeof(int)));
  TEST_HIP_CHECK(hipMalloc(&d_col_index, nnz * sizeof(int)));
  TEST_HIP_CHECK(hipMalloc(&d_values, nnz * sizeof(double)));
  TEST_HIP_CHECK(hipMalloc(&d_x, N * sizeof(double)));
  TEST_HIP_CHECK(hipMalloc(&d_y, N * sizeof(double)));

  TEST_HIP_CHECK(hipMemcpy(d_row_ptr, h_row_ptr, (N + 1) * sizeof(int), hipMemcpyHostToDevice));
  TEST_HIP_CHECK(hipMemcpy(d_col_index, h_col_index, nnz * sizeof(int), hipMemcpyHostToDevice));
  TEST_HIP_CHECK(hipMemcpy(d_values, h_values, nnz * sizeof(double), hipMemcpyHostToDevice));
  TEST_HIP_CHECK(hipMemcpy(d_x, h_x, N * sizeof(double), hipMemcpyHostToDevice));
  TEST_HIP_CHECK(hipMemset(d_y, 0, N * sizeof(double)));

  csr_desc<int, double> csr_desc(N, N, nnz, d_row_ptr, d_col_index, d_values);

  spmv_vector_row(0, 1.0, 0.0, csr_desc, d_x, d_y);
  hipDeviceSynchronize();

  TEST_HIP_CHECK(hipMemcpy(h_y, d_y, N * sizeof(double), hipMemcpyDeviceToHost));

  for (int i = 0; i < N; i++) {
    if (h_y_ref[i] != 0.0) {
      double relative_error = std::abs(h_y[i] - h_y_ref[i]) / h_y_ref[i];
      EXPECT_NEAR(relative_error, 0.0, 1e-12);
    } else {
      EXPECT_NEAR(h_y[i], 0.0, 1e-12);
    }
  }

  TEST_HIP_CHECK(hipFree(d_row_ptr));
  TEST_HIP_CHECK(hipFree(d_col_index));
  TEST_HIP_CHECK(hipFree(d_values));
  TEST_HIP_CHECK(hipFree(d_x));
  TEST_HIP_CHECK(hipFree(d_y));

  delete[] h_x;
  delete[] h_y;
  delete[] h_y_ref;
  delete[] h_row_ptr;
  delete[] h_col_index;
  delete[] h_values;
}

TEST(ustbSparseTest, spmv_line_enhance) {
  constexpr int N = 100000;
  constexpr int nnz_per_row = 20;
  constexpr int nnz = N * nnz_per_row;

  double *h_x = new double[N];
  double *h_y = new double[N];
  double *h_y_ref = new double[N];
  int *h_row_ptr = new int[N + 1];
  int *h_col_index = new int[nnz];
  double *h_values = new double[nnz];

  for (int i = 0; i < N; i++) {
    h_row_ptr[i] = i * nnz_per_row;
    h_x[i] = 1.0 + (i % 7);

    for (int j = 0; j < nnz_per_row; j++) {
      int col = i - 2 + j;
      if (col < 0)
        col += N;
      if (col >= N)
        col -= N;
      h_col_index[i * nnz_per_row + j] = col;
      h_values[i * nnz_per_row + j] = (i == col) ? 4.0 : 1.0;
    }
  }
  h_row_ptr[N] = nnz;

  std::fill(h_y_ref, h_y_ref + N, 0.0);
  for (int i = 0; i < N; i++) {
    for (int j = h_row_ptr[i]; j < h_row_ptr[i + 1]; j++) {
      h_y_ref[i] += h_values[j] * h_x[h_col_index[j]];
    }
  }

  int *d_row_ptr = nullptr;
  int *d_col_index = nullptr;
  double *d_values = nullptr;
  double *d_x = nullptr;
  double *d_y = nullptr;

  TEST_HIP_CHECK(hipMalloc(&d_row_ptr, (N + 1) * sizeof(int)));
  TEST_HIP_CHECK(hipMalloc(&d_col_index, nnz * sizeof(int)));
  TEST_HIP_CHECK(hipMalloc(&d_values, nnz * sizeof(double)));
  TEST_HIP_CHECK(hipMalloc(&d_x, N * sizeof(double)));
  TEST_HIP_CHECK(hipMalloc(&d_y, N * sizeof(double)));

  TEST_HIP_CHECK(hipMemcpy(d_row_ptr, h_row_ptr, (N + 1) * sizeof(int), hipMemcpyHostToDevice));
  TEST_HIP_CHECK(hipMemcpy(d_col_index, h_col_index, nnz * sizeof(int), hipMemcpyHostToDevice));
  TEST_HIP_CHECK(hipMemcpy(d_values, h_values, nnz * sizeof(double), hipMemcpyHostToDevice));
  TEST_HIP_CHECK(hipMemcpy(d_x, h_x, N * sizeof(double), hipMemcpyHostToDevice));
  TEST_HIP_CHECK(hipMemset(d_y, 0, N * sizeof(double)));

  csr_desc<int, double> csr_desc(N, N, nnz, d_row_ptr, d_col_index, d_values);

  spmv_line_enhance(0, 1.0, 0.0, csr_desc, d_x, d_y);
  hipDeviceSynchronize();

  TEST_HIP_CHECK(hipMemcpy(h_y, d_y, N * sizeof(double), hipMemcpyDeviceToHost));

  for (int i = 0; i < N; i++) {
    if (h_y_ref[i] != 0.0) {
      double relative_error = std::abs(h_y[i] - h_y_ref[i]) / h_y_ref[i];
      EXPECT_NEAR(relative_error, 0.0, 1e-12);
    } else {
      EXPECT_NEAR(h_y[i], 0.0, 1e-12);
    }
  }

  TEST_HIP_CHECK(hipFree(d_row_ptr));
  TEST_HIP_CHECK(hipFree(d_col_index));
  TEST_HIP_CHECK(hipFree(d_values));
  TEST_HIP_CHECK(hipFree(d_x));
  TEST_HIP_CHECK(hipFree(d_y));

  delete[] h_x;
  delete[] h_y;
  delete[] h_y_ref;
  delete[] h_row_ptr;
  delete[] h_col_index;
  delete[] h_values;
}

TEST(ustbSparseTest, spmv_flat) {
  constexpr int N = 100000;
  constexpr int nnz_per_row = 20;
  constexpr int nnz = N * nnz_per_row;

  double *h_x = new double[N];
  double *h_y = new double[N];
  double *h_y_ref = new double[N];
  int *h_row_ptr = new int[N + 1];
  int *h_col_index = new int[nnz];
  double *h_values = new double[nnz];

  for (int i = 0; i < N; i++) {
    h_row_ptr[i] = i * nnz_per_row;
    h_x[i] = 1.0 + (i % 7);

    for (int j = 0; j < nnz_per_row; j++) {
      int col = i - 2 + j;
      if (col < 0)
        col += N;
      if (col >= N)
        col -= N;
      h_col_index[i * nnz_per_row + j] = col;
      h_values[i * nnz_per_row + j] = (i == col) ? 4.0 : 1.0;
    }
  }
  h_row_ptr[N] = nnz;

  std::fill(h_y_ref, h_y_ref + N, 0.0);
  for (int i = 0; i < N; i++) {
    for (int j = h_row_ptr[i]; j < h_row_ptr[i + 1]; j++) {
      h_y_ref[i] += h_values[j] * h_x[h_col_index[j]];
    }
  }

  int *d_row_ptr = nullptr;
  int *d_col_index = nullptr;
  double *d_values = nullptr;
  double *d_x = nullptr;
  double *d_y = nullptr;

  TEST_HIP_CHECK(hipMalloc(&d_row_ptr, (N + 1) * sizeof(int)));
  TEST_HIP_CHECK(hipMalloc(&d_col_index, nnz * sizeof(int)));
  TEST_HIP_CHECK(hipMalloc(&d_values, nnz * sizeof(double)));
  TEST_HIP_CHECK(hipMalloc(&d_x, N * sizeof(double)));
  TEST_HIP_CHECK(hipMalloc(&d_y, N * sizeof(double)));

  TEST_HIP_CHECK(hipMemcpy(d_row_ptr, h_row_ptr, (N + 1) * sizeof(int), hipMemcpyHostToDevice));
  TEST_HIP_CHECK(hipMemcpy(d_col_index, h_col_index, nnz * sizeof(int), hipMemcpyHostToDevice));
  TEST_HIP_CHECK(hipMemcpy(d_values, h_values, nnz * sizeof(double), hipMemcpyHostToDevice));
  TEST_HIP_CHECK(hipMemcpy(d_x, h_x, N * sizeof(double), hipMemcpyHostToDevice));
  TEST_HIP_CHECK(hipMemset(d_y, 0, N * sizeof(double)));

  csr_desc<int, double> h_csr_desc(N, N, nnz, h_row_ptr, h_col_index, h_values);
  csr_desc<int, double> d_csr_desc(N, N, nnz, d_row_ptr, d_col_index, d_values);

  spmv_flat(0, 1.0, 0.0, h_csr_desc, d_csr_desc, d_x, d_y);
  hipDeviceSynchronize();

  TEST_HIP_CHECK(hipMemcpy(h_y, d_y, N * sizeof(double), hipMemcpyDeviceToHost));

  for (int i = 0; i < N; i++) {
    if (h_y_ref[i] != 0.0) {
      double relative_error = std::abs(h_y[i] - h_y_ref[i]) / h_y_ref[i];
      EXPECT_NEAR(relative_error, 0.0, 1e-12);
    } else {
      EXPECT_NEAR(h_y[i], 0.0, 1e-12);
    }
  }

  TEST_HIP_CHECK(hipFree(d_row_ptr));
  TEST_HIP_CHECK(hipFree(d_col_index));
  TEST_HIP_CHECK(hipFree(d_values));
  TEST_HIP_CHECK(hipFree(d_x));
  TEST_HIP_CHECK(hipFree(d_y));

  delete[] h_x;
  delete[] h_y;
  delete[] h_y_ref;
  delete[] h_row_ptr;
  delete[] h_col_index;
  delete[] h_values;
}
