#include <hip/hip_runtime.h>

#include "level3/csrsm_kernels.h"

#include "common.h"
#include "common/usparse_utils.hpp"

template <typename T>
T *test_updown_matrix(const MatrixData<T> &matrix_in, const T *h_b, int k, int h_rhs, int *n_out, bool is_lower,
                      float *out_time) {
  int j = k;
  const int n = matrix_in.rows;
  const int rhs = h_rhs;
  const int nnz = matrix_in.nnz;
  *n_out = n;
  // 使用vector的data()获取原始指针
  const int *h_row_ptr = matrix_in.row_ptr.data();
  const int *h_col_idx = matrix_in.col_index.data();
  const T *h_val = reinterpret_cast<const T *>(matrix_in.values.data());

  // 使用动态数组替换VLA
  T *h_x = new T[n * rhs]();
  int *h_is_solved = new int[n * rhs]();
  T *h_temp = new T[n]();

  // 设备内存分配
  int *d_row_ptr, *d_col_idx, *d_is_solved;
  T *d_val, *d_b, *d_x, *d_temp;
  USPARSE_HIP_API_CHECK(hipMalloc(&d_row_ptr, (n + 1) * sizeof(int)));
  USPARSE_HIP_API_CHECK(hipMalloc(&d_col_idx, nnz * sizeof(int)));
  USPARSE_HIP_API_CHECK(hipMalloc(&d_val, nnz * sizeof(T)));
  USPARSE_HIP_API_CHECK(hipMalloc(&d_b, n * rhs * sizeof(T)));
  USPARSE_HIP_API_CHECK(hipMalloc(&d_x, n * rhs * sizeof(T)));
  USPARSE_HIP_API_CHECK(hipMalloc(&d_is_solved, n * rhs * sizeof(int)));
  USPARSE_HIP_API_CHECK(hipMalloc(&d_temp, n * sizeof(T)));

  // 数据拷贝到设备
  USPARSE_HIP_API_CHECK(hipMemcpy(d_row_ptr, h_row_ptr, (n + 1) * sizeof(int), hipMemcpyHostToDevice));
  USPARSE_HIP_API_CHECK(hipMemcpy(d_col_idx, h_col_idx, nnz * sizeof(int), hipMemcpyHostToDevice));
  USPARSE_HIP_API_CHECK(hipMemcpy(d_val, h_val, nnz * sizeof(T), hipMemcpyHostToDevice));
  USPARSE_HIP_API_CHECK(hipMemcpy(d_b, h_b, n * rhs * sizeof(T), hipMemcpyHostToDevice));
  USPARSE_HIP_API_CHECK(hipMemcpy(d_x, h_x, n * rhs * sizeof(T), hipMemcpyHostToDevice));
  USPARSE_HIP_API_CHECK(hipMemcpy(d_is_solved, h_is_solved, n * rhs * sizeof(int), hipMemcpyHostToDevice));
  USPARSE_HIP_API_CHECK(hipMemcpy(d_temp, h_temp, n * sizeof(T), hipMemcpyHostToDevice));

  // Kernel 调用
  int blockSize = 256;                                         // Block 的大小
  int numBlocks = (n * WARP_SIZE + blockSize - 1) / blockSize; // Block 数
  // 预热
  for (int i = 0; i < 5; i++)
    hipLaunchKernelGGL(sptrsm_updown_kernel<T>, dim3(numBlocks), dim3(blockSize), blockSize * sizeof(T), 0, d_row_ptr,
                       d_col_idx, d_val, d_b, d_x, d_is_solved, n, rhs, is_lower); // kernel
  // 开始计时
  // my_timer tm;
  // tm.start();
  double time1 = 0.0;
  for (int i = 0; i < j; i++) {
    USPARSE_HIP_API_CHECK(hipMemset(d_is_solved, 0, n * rhs * sizeof(int)));
    // 开始计时
    my_timer tm;
    tm.start();
    hipLaunchKernelGGL(sptrsm_updown_kernel<T>, dim3(numBlocks), dim3(blockSize), blockSize * sizeof(T), 0, d_row_ptr,
                       d_col_idx, d_val, d_b, d_x, d_is_solved, n, rhs, is_lower); // kernel
    USPARSE_HIP_API_CHECK(hipDeviceSynchronize());
    tm.stop(); // 结束计时
    time1 += tm.time_use;
    if (i == 0)
      USPARSE_HIP_API_CHECK(hipMemcpy(h_x, d_x, n * rhs * sizeof(T), hipMemcpyDeviceToHost));
  }
  // tm.stop();//结束计时
  // std::cout << "GPU rocsparse computation time: " << tm.time_use/k *1.0 << " us" << std::endl;
  *out_time = time1 / k * 1.0;

  // 数据拷贝回主机
  // USPARSE_HIP_API_CHECK(hipMemcpy(h_x, d_x, n * rhs * sizeof(T), hipMemcpyDeviceToHost));
  USPARSE_HIP_API_CHECK(hipMemcpy(h_is_solved, d_is_solved, n * rhs * sizeof(int), hipMemcpyDeviceToHost));
  USPARSE_HIP_API_CHECK(hipMemcpy(h_temp, d_temp, n * sizeof(T), hipMemcpyDeviceToHost));

  // 拷贝结果到新数组并释放原内存
  T *result = new T[n * rhs];
  memcpy(result, h_x, n * rhs * sizeof(T));

  // 释放主机内存
  delete[] h_x;
  delete[] h_is_solved;
  delete[] h_temp;

  // 释放设备内存
  USPARSE_HIP_API_CHECK(hipFree(d_row_ptr));
  USPARSE_HIP_API_CHECK(hipFree(d_col_idx));
  USPARSE_HIP_API_CHECK(hipFree(d_val));
  USPARSE_HIP_API_CHECK(hipFree(d_b));
  USPARSE_HIP_API_CHECK(hipFree(d_x));
  USPARSE_HIP_API_CHECK(hipFree(d_is_solved));
  USPARSE_HIP_API_CHECK(hipFree(d_temp));

  return result;
}

// 显式实例化
template float *test_updown_matrix<float>(const MatrixData<float> &, const float *, int, int, int *, bool, float *);
template double *test_updown_matrix<double>(const MatrixData<double> &, const double *, int, int, int *, bool, float *);
