#include "gmres.hpp"
#include "sparseMatrix.hpp"
#include <assert.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <tuple>
#include <vector>

#include <hip/hip_runtime.h>

#include "gmres_optimization_config.h"
#include "gmres_utils.hpp"
#include "init.hpp"
#include "usparse_gmres.h"

using namespace std;

const int RESTART_TIMES = 20;        // 禁止修改
const double REL_RESID_LIMIT = 1e-6; // 禁止修改
const int ITERATION_LIMIT = 10000;   // 禁止修改

double calculateNorm(const double *vec, uint N) {
  double sum = 0.0;
  for (uint i = 0; i < N; ++i) {
    sum += vec[i] * vec[i];
  }
  return std::sqrt(sum);
}

void spmv(const uint *rowPtr, const uint *colInd, const double *values, const double *x, double *y, uint numRows) {
  for (uint i = 0; i < numRows; ++i) {
    double sum = 0.0;
    for (uint j = rowPtr[i]; j < rowPtr[i + 1]; ++j) {
      sum += values[j] * x[colInd[j]];
    }
    y[i] = sum;
  }
}

template <typename T>
usparse::usparse_csrmv_algo select_csrmv_algo(const csr_desc<int, T> d_csr_desc) {
  const int density = d_csr_desc.nnz / d_csr_desc.rows;

  if (density > 50.0) {
    if (d_csr_desc.nnz > 10000000 && d_csr_desc.rows > 1000000) {
      return usparse::usparse_csrmv_algo_line_enhance;
    } else {
      return usparse::usparse_csrmv_algo_flat;
    }
  } else if (density > 15.0) {
    if (d_csr_desc.nnz > 5000000) {
      return usparse::usparse_csrmv_algo_flat;
    } else if (d_csr_desc.rows > 500000) {
      return usparse::usparse_csrmv_algo_line_enhance;
    } else {
      return usparse::usparse_csrmv_algo_flat;
    }
  } else {
    if (d_csr_desc.nnz < 1000000 || d_csr_desc.rows > 1000000) {
      return usparse::usparse_csrmv_algo_line_enhance;
    } else {
      if (density > 8.0) {
        return usparse::usparse_csrmv_algo_flat;
      } else {
        return usparse::usparse_csrmv_algo_line_enhance;
      }
    }
  }
}

struct sync_handle {
  static constexpr int SYNC_POLICY = 2;

  static constexpr int SYNC_POLICY_DEVICE_SYNC = 0;
  static constexpr int SYNC_POLICY_EVENT_SYNC = 1;
  static constexpr int SYNC_POLICY_MEMCOPY_SYNC = 2;

  inline sync_handle() {
    if (SYNC_POLICY == SYNC_POLICY_DEVICE_SYNC) {
      return;
    } else if (SYNC_POLICY == SYNC_POLICY_EVENT_SYNC) {
      hipEventCreate(&sync_kernel_event);
    } else if (SYNC_POLICY == SYNC_POLICY_MEMCOPY_SYNC) {
      mem_malloc();
    }
  }

  inline void sync() {
    if (SYNC_POLICY == SYNC_POLICY_DEVICE_SYNC) {
      hipDeviceSynchronize();
    } else if (SYNC_POLICY == SYNC_POLICY_EVENT_SYNC) {
      hipEventRecord(sync_kernel_event);
      hipEventSynchronize(sync_kernel_event);
    } else if (SYNC_POLICY == SYNC_POLICY_MEMCOPY_SYNC) {
      mem_copy();
    }
  }

private:
  int *host_data;
  int *dev_data;

  hipEvent_t sync_kernel_event;

private:
  void mem_malloc() {
    HIP_CHECK(hipHostMalloc((int **)&host_data, 1 * sizeof(int)));
    HIP_CHECK(hipMalloc((int **)&dev_data, 1 * sizeof(int)));
  }

  void mem_copy() { HIP_CHECK(hipMemcpy(dev_data, host_data, 1 * sizeof(int), hipMemcpyHostToDevice)); }
};

RESULT gmres(SpM<double> *A_d, double *x_d, double *_b, SpM<double> *dev_A, double *dev_x, double *dev_y,
             csr_desc<int, double> d_csr_desc) {
  // 稀疏矩阵A_d需默认以CSR格式压缩存储，使用DCU优化时，该矩阵A和右端向量b传输到DCU端的开销，以及结果向量x传输到CPU端的开销可不算在计时范围
  // 若要采用其他的稀疏矩阵压缩格式，需从DCU端的CSR矩阵开始转换，且格式转换的代码必须包含到计时范围内
  const uint N = A_d->nrows;

  std::vector<double> r0(N);
  std::vector<double> V((RESTART_TIMES + 1) * N);
  std::vector<double> s(RESTART_TIMES + 1, 0.0);
  std::vector<double> V0(N);
  std::vector<double> H((RESTART_TIMES + 1) * RESTART_TIMES);
  std::vector<double> cs(RESTART_TIMES);
  std::vector<double> sn(RESTART_TIMES);

  double H_cpu;
  double beta_cpu;

  double alpha;

  double beta;
  beta = calculateNorm(_b, N);                 // 禁止修改
  double RESID_LIMIT = REL_RESID_LIMIT * beta; // 禁止修改
  double init_res = beta;                      // 禁止修改

  int i, j, k;
  double resid;
  double tmp;
  int iteration = 0;

  // Apply for memory on the device
  double *d_r0 = nullptr;
  double *d_H = nullptr;
  double *d_V = nullptr;
  double *d_s = nullptr;
  double *d_cs = nullptr;
  double *d_sn = nullptr;
  double *d_vi = nullptr;
  double *d_beta = nullptr;
  double *d_b = nullptr;
  double *d_temp_buffer = nullptr;
  init_device_matrix(&d_r0, &d_V, &d_s, &d_H, &d_cs, &d_sn, &d_vi, N, RESTART_TIMES);
  // std::cout << "d_r0 =" << d_r0 << std::endl;
  // std::cout << "d_H =" << d_H << std::endl;
  // std::cout << "d_V =" << d_V << std::endl;
  // std::cout << "d_s =" << d_s << std::endl;
  // std::cout << "d_cs =" << d_cs << std::endl;
  // std::cout << "d_sn =" << d_sn << std::endl;
  // std::cout << "d_vi =" << d_vi << std::endl;

  HIP_CHECK(hipMalloc(&d_beta, sizeof(double)));
  HIP_CHECK(hipMalloc(&d_b, N * sizeof(double)));
  HIP_CHECK(hipMalloc(&d_temp_buffer, (N / 256 + (N % 256 == 0 ? 0 : 1)) * sizeof(double)));
  HIP_CHECK(hipMemcpy(dev_x, x_d, N * sizeof(double), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b, _b, N * sizeof(double), hipMemcpyHostToDevice));

  usparse::csrmv_mat_info csrmv_info;
  usparse::device_adaptive_analysis(0, 1.0, 0.0, d_csr_desc, usparse::usparse_csrmv_algo_flat, &csrmv_info);
  utils::warmup();

  // auto start = std::chrono::high_resolution_clock::now(); // 禁止修改
  // =======如果已移植为DCU代码，请更换为下列计时==========
  hipEvent_t test_start_event, test_stop_event;
  hipEventCreate(&test_start_event);
  hipEventCreate(&test_stop_event);
  hipEventRecord(test_start_event, 0);

  // 任何对稀疏矩阵的预处理操作，如稀疏矩阵压缩格式转换、非零元数组或向量的精度转换、稀疏矩阵的非零元特征计算等，均需放在计时范围内，相关内存申请和释放除外

  /****GMRES计算过程****/
  do {
    // ==========外迭代============
    // spmv(A_d->rows, A_d->cols, A_d->vals, x_d, r0.data(), N); // 不可修改此步操作中相关数据的存储精度和SpMV计算精度
    // copy x from host to device

    // compute y = AX
    device_adaptive_spmv<int, double, BEAT_Y_POLICY_ZERO>(0, 1.0, 0.0, d_csr_desc, dev_x, dev_y,
                                                          usparse::usparse_csrmv_algo_flat, &csrmv_info);

    usparse::device_daxpy(-1.0, d_b, dev_y, N);

    usparse::device_calculateNorm(dev_y, N, d_beta, d_temp_buffer, 0);
    hipDeviceSynchronize();

    beta = *d_beta;

    usparse::device_dscal(-1.0 / beta, dev_y, N);

    usparse::device_dcopy(dev_y, d_V, N);

    HIP_CHECK(hipMemset(d_s, 0.0, (RESTART_TIMES + 1) * sizeof(double)));
    HIP_CHECK(hipMemcpy(&d_s[0], d_beta, sizeof(double), hipMemcpyDeviceToDevice));

    resid = std::abs(beta);
    i = -1;

    if (resid <= RESID_LIMIT || iteration >= ITERATION_LIMIT) {
      break;
    }
    do {
      // ==========内迭代============
      i++;
      iteration++;

      std::vector<double> V_i(N);
      usparse::device_dcopy(d_V + i * N, d_vi, N);

      usparse::device_adaptive_spmv<int, double, BEAT_Y_POLICY_ZERO>(0, 1.0, 0.0, d_csr_desc, d_vi, dev_y,
                                                                     usparse::usparse_csrmv_algo_flat, &csrmv_info);

      for (k = 0; k <= i; k++) {
        usparse::device_dotProduct(dev_y, d_V + k * N, N, &d_H[k * RESTART_TIMES + i], d_temp_buffer, 0);
        hipDeviceSynchronize();
        alpha = -d_H[k * RESTART_TIMES + i];
        usparse::device_daxpy(alpha, d_V + N * k, dev_y, N);
      }
      usparse::device_calculateNorm(dev_y, N, &d_H[(i + 1) * RESTART_TIMES + i], d_temp_buffer, 0);
      hipDeviceSynchronize();

      alpha = 1.0 / d_H[(i + 1) * RESTART_TIMES + i];
      usparse::device_dscal(alpha, dev_y, N);
      usparse::device_dcopy(dev_y, d_V + N * (i + 1), N);

      usparse::device_rotation2((uint)(RESTART_TIMES), d_H, d_cs, d_sn, d_s, (uint)(i)); // todo: uint to int
      hipDeviceSynchronize();
      tmp = d_s[i + 1];
      resid = std::abs(tmp);
      // std::cout << "iteration " << iteration << ", resid = " << resid/init_res << std::endl;

      if (resid <= RESID_LIMIT || iteration >= ITERATION_LIMIT) {
        break;
      }
    } while (i + 1 < RESTART_TIMES && iteration <= ITERATION_LIMIT);

    // 求解上三角系统
    usparse::device_sovlerTri(RESTART_TIMES, i, d_H, d_s);
    hipDeviceSynchronize();

    // 更新解
    for (j = 0; j <= i; j++) {
      tmp = d_s[j];
      usparse::device_daxpy(tmp, d_V + j * N, dev_x, N);
    }
  } while (resid > RESID_LIMIT && iteration <= ITERATION_LIMIT);

  // auto stop = std::chrono::high_resolution_clock::now();            // 禁止修改
  // std::chrono::duration<float, std::milli> duration = stop - start; // 禁止修改
  // float test_time = duration.count();                               // 禁止修改
  // =======如果已移植为DCU代码，请更换为下列计时==========
  hipEventRecord(test_stop_event, 0);                                 // 禁止修改
  hipEventSynchronize(test_stop_event);                               // 禁止修改
  float test_time = 0.0;                                              // 禁止修改
  hipEventElapsedTime(&test_time, test_start_event, test_stop_event); // 禁止修改

  // destory data
  destory_device_matrix(d_r0, d_V, d_s, d_H, d_cs, d_sn, d_vi);

  HIP_CHECK(hipFree(d_beta));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_temp_buffer));
  // todo: destroy csrmv info.
  return make_tuple(iteration, test_time, resid / init_res); // 禁止修改
}

RESULT gmres_float(SpM<double> *A_d, double *x_d, double *_b, SpM<double> *dev_A, double *dev_x, double *dev_y,
                   csr_desc<int, double> d_csr_desc) {
  // 稀疏矩阵A_d需默认以CSR格式压缩存储，使用DCU优化时，该矩阵A和右端向量b传输到DCU端的开销，以及结果向量x传输到CPU端的开销可不算在计时范围
  // 若要采用其他的稀疏矩阵压缩格式，需从DCU端的CSR矩阵开始转换，且格式转换的代码必须包含到计时范围内
  const uint N = A_d->nrows;

  std::vector<double> r0(N);
  std::vector<double> V((RESTART_TIMES + 1) * N);
  std::vector<double> s(RESTART_TIMES + 1, 0.0);
  std::vector<double> V0(N);
  std::vector<double> H((RESTART_TIMES + 1) * RESTART_TIMES);
  std::vector<double> cs(RESTART_TIMES);
  std::vector<double> sn(RESTART_TIMES);

  double H_cpu;
  double beta_cpu;

  double alpha;

  double beta;
  beta = calculateNorm(_b, N);                 // 禁止修改
  double RESID_LIMIT = REL_RESID_LIMIT * beta; // 禁止修改
  double init_res = beta;                      // 禁止修改

  int i, j, k;
  double resid;
  double tmp;
  int iteration = 0;

  // Apply for memory on the device
  double *d_r0 = nullptr;
  double *d_H = nullptr;
  double *d_V = nullptr;
  double *d_s = nullptr;
  double *d_cs = nullptr;
  double *d_sn = nullptr;
  double *d_vi = nullptr;
  // double *d_beta = nullptr;
  double *d_b = nullptr;
  float *d_temp_buffer = nullptr;
  float *d_gs_buffer = nullptr;
  init_device_matrix(&d_r0, &d_V, &d_s, &d_H, &d_cs, &d_sn, &d_vi, N, RESTART_TIMES);

  //  float var
  float *d_H_f = nullptr;
  float *d_V_f = nullptr;
  float *d_vi_f = nullptr;
  float *d_sn_f = nullptr;
  float *d_cs_f = nullptr;
  float *d_r0_f = nullptr;    // dev_y: r0
  float *d_A_val_f = nullptr; // A: float
  float *d_s_f = nullptr;
  float *d_b_f = nullptr;
  float *d_beta_f = nullptr;
  // apply for float var
  init_device_matrix_float(&d_V_f, &d_H_f, &d_vi_f, &d_sn_f, &d_cs_f, &d_r0_f, &d_A_val_f, &d_s_f, &d_b_f, N,
                           RESTART_TIMES, dev_A->nnz);

  HIP_CHECK(hipMalloc(&d_beta_f, sizeof(float)));
  HIP_CHECK(hipMalloc(&d_b, N * sizeof(double)));
  HIP_CHECK(hipMalloc(&d_temp_buffer, (N / 256 + (N % 256 == 0 ? 0 : 1)) * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gs_buffer, RESTART_TIMES * (N / 256 + (N % 256 == 0 ? 0 : 1)) * sizeof(float)));
  HIP_CHECK(hipMemcpy(dev_x, x_d, N * sizeof(double), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b, _b, N * sizeof(double), hipMemcpyHostToDevice));

  // force conversion precision form dev_A.val to d_A_val_f
  usparse::device_cast(dev_A->vals, d_A_val_f, dev_A->nnz);
  usparse::device_cast(d_b, d_b_f, N);

  // initialize info descrA handle
  csr_desc<int, float> d_csr_desc_float(dev_A->nrows, dev_A->ncols, d_csr_desc.nnz, d_csr_desc.row_ptr,
                                        d_csr_desc.col_index, d_A_val_f);
  usparse::csrmv_mat_info csrmv_info;
  const usparse::usparse_csrmv_algo csrmv_algo = select_csrmv_algo(d_csr_desc);
  usparse::device_adaptive_analysis(0, 1.0, 0.0, d_csr_desc, csrmv_algo, &csrmv_info);

  utils::warmup();

  // auto start = std::chrono::high_resolution_clock::now(); // 禁止修改
  // =======如果已移植为DCU代码，请更换为下列计时==========
  hipEvent_t test_start_event, test_stop_event;
  sync_handle h_sync;
  hipEventCreate(&test_start_event);
  hipEventCreate(&test_stop_event);
  hipEventRecord(test_start_event, 0);

  // 任何对稀疏矩阵的预处理操作，如稀疏矩阵压缩格式转换、非零元数组或向量的精度转换、稀疏矩阵的非零元特征计算等，均需放在计时范围内，相关内存申请和释放除外

  /****GMRES计算过程****/
  do {
    // ==========外迭代============
    // spmv(A_d->rows, A_d->cols, A_d->vals, x_d, r0.data(), N); // 不可修改此步操作中相关数据的存储精度和SpMV计算精度
    // copy x from host to device

    // if it is the first time, dev_x is 0. Ax is also 0.
    if(iteration != 0) {
      // compute y = Ax
      usparse::device_adaptive_spmv<int, double, BEAT_Y_POLICY_ZERO>(0, 1.0, 0.0, d_csr_desc, dev_x, dev_y, csrmv_algo, &csrmv_info);

      usparse::device_cast<uint, double, float>(dev_y, d_r0_f, N); // dev_y => d_r0_f

      // float
      usparse::device_daxpy<uint, float, float>(-1.0, d_b_f, d_r0_f, N); // d_r0_f += -1.0 * d_b_f.

      usparse::device_calculateNorm<uint, float>(d_r0_f, N, d_beta_f, d_temp_buffer, 0);
      h_sync.sync();

      beta = *d_beta_f;
    } else {
      // if it is the first iteration. the above code in if branch can be simplified as following.
      // d_r0_f = -d_b_f
      usparse::device_copy_v2(d_b_f, d_r0_f, (float)(-1.0), N);
      // d_beta_f = norm(-d_b_f); which is also beta.
      *d_beta_f = (float) beta;
    }

    usparse::device_dscal<uint, float>(-1.0 / beta, d_r0_f, N);

    usparse::device_dcopy<uint, float>(d_r0_f, d_V_f, N);

    usparse::device_memset<uint,float>(d_s_f, (RESTART_TIMES + 1));
    HIP_CHECK(hipMemcpy(&d_s_f[0], d_beta_f, sizeof(float), hipMemcpyDeviceToDevice));

    resid = std::abs(beta);
    i = -1;

    if (resid <= RESID_LIMIT || iteration >= ITERATION_LIMIT) {
      break;
    }
    do {
      // ==========内迭代============
      i++;
      iteration++;

      const float *csrmv_x_f = d_V_f + i * N; // the x vector in y = Ax.

      usparse::device_adaptive_spmv<int, float, BEAT_Y_POLICY_ZERO>(0, 1.0, 0.0, d_csr_desc_float, csrmv_x_f, d_r0_f,
                                                                    csrmv_algo, &csrmv_info);

      if (usparse::GMRES_ORTHO_USE == usparse::GMRES_OPT_USE_MGS) {
        if (usparse::GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION ==
usparse::GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION_LEVEL_2) {
          usparse::dotprod_axpy_fusion<uint, float, usparse::GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION_LEVEL_2>(
                d_r0_f, d_V_f, d_H_f, d_temp_buffer, N, RESTART_TIMES, i, 0);
        } else if (usparse::GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION ==
                 usparse::GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION_LEVEL_1) {
          usparse::dotprod_axpy_fusion<uint, float, usparse::GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION_LEVEL_1>(
                d_r0_f, d_V_f, d_H_f, d_temp_buffer, N, RESTART_TIMES, i, 0);
          usparse::device_calculateNorm(d_r0_f, N, &d_H_f[(i + 1) * RESTART_TIMES + i], d_temp_buffer, 0);
          h_sync.sync();
          alpha = 1.0 / d_H_f[(i + 1) * RESTART_TIMES + i];
          usparse::device_dscal<uint, float>(alpha, d_r0_f, N);
          usparse::device_dcopy(d_r0_f, d_V_f + N * (i + 1), N);
        } else { // GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION_NONE
          for (k = 0; k <= i; k++) {
            usparse::device_dotProduct(d_r0_f, d_V_f + k * N, N, &d_H_f[k * RESTART_TIMES + i], d_temp_buffer, 0);
            h_sync.sync();
            alpha = -d_H_f[k * RESTART_TIMES + i];
            usparse::device_daxpy<uint, float, float>(alpha, d_V_f + N * k, d_r0_f, N);
          }
          usparse::device_calculateNorm(d_r0_f, N, &d_H_f[(i + 1) * RESTART_TIMES + i], d_temp_buffer, 0);
          h_sync.sync();

          alpha = 1.0 / d_H_f[(i + 1) * RESTART_TIMES + i];
          usparse::device_dscal<uint, float>(alpha, d_r0_f, N);
          usparse::device_dcopy(d_r0_f, d_V_f + N * (i + 1), N);
        }
      } else { // GMRES_OPT_USE_GS
        if (usparse::GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION ==
            usparse::GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION_LEVEL_2) {
          usparse::gs_fusion<uint, float, usparse::GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION_LEVEL_2>(
              d_r0_f, d_V_f, d_H_f, d_gs_buffer, N, RESTART_TIMES, i, 0);
        } else if (usparse::GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION ==
                   usparse::GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION_LEVEL_1) {
          usparse::gs_dotproduct<uint, float>(d_r0_f, d_V_f, d_H_f, d_gs_buffer, N, (uint)i, 0);

          h_sync.sync();

          usparse::gs_daxpy<uint, float>(d_r0_f, d_V_f, d_H_f, N, (uint)i);
          usparse::device_calculateNorm(d_r0_f, N, &d_H_f[(i + 1) * RESTART_TIMES + i], d_temp_buffer, 0);
          h_sync.sync();

          alpha = 1.0 / d_H_f[(i + 1) * RESTART_TIMES + i];
          usparse::device_dscal<uint, float>(alpha, d_r0_f, N);
          usparse::device_dcopy(d_r0_f, d_V_f + N * (i + 1), N);
        } else {
          for (k = 0; k <= i; k++) {
            usparse::device_dotProduct(d_r0_f, d_V_f + k * N, N, &d_H_f[k * RESTART_TIMES + i], d_temp_buffer, 0);
          }
          h_sync.sync();
          for (k = 0; k <= i; k++) {
            alpha = -d_H_f[k * RESTART_TIMES + i];
            usparse::device_daxpy<uint, float, float>(alpha, d_V_f + N * k, d_r0_f, N);
          }
          usparse::device_calculateNorm(d_r0_f, N, &d_H_f[(i + 1) * RESTART_TIMES + i], d_temp_buffer, 0);
          h_sync.sync();

          alpha = 1.0 / d_H_f[(i + 1) * RESTART_TIMES + i];
          usparse::device_dscal<uint, float>(alpha, d_r0_f, N);
          usparse::device_dcopy(d_r0_f, d_V_f + N * (i + 1), N);
        }
      }

      usparse::device_rotation2((uint)(RESTART_TIMES), d_H_f, d_cs_f, d_sn_f, d_s_f, (uint)(i)); // todo: uint to int
      h_sync.sync();
      tmp = d_s_f[i + 1];
      resid = std::abs(tmp);
      // std::cout << "iteration " << iteration << ", resid = " << resid/init_res << std::endl;

      if (resid <= RESID_LIMIT || iteration >= ITERATION_LIMIT) {
        break;
      }
    } while (i + 1 < RESTART_TIMES && iteration <= ITERATION_LIMIT);

    // 求解上三角系统
    usparse::device_sovlerTri(RESTART_TIMES, i, d_H_f, d_s_f);
    h_sync.sync();

    // 更新解: <float, double>
    for (j = 0; j <= i; j++) {
      tmp = d_s_f[j];
      usparse::device_daxpy<uint, float, double>(tmp, d_V_f + j * N, dev_x, N);
    }
  } while (resid > RESID_LIMIT && iteration <= ITERATION_LIMIT);

  // auto stop = std::chrono::high_resolution_clock::now();            // 禁止修改
  // std::chrono::duration<float, std::milli> duration = stop - start; // 禁止修改
  // float test_time = duration.count();                               // 禁止修改
  // =======如果已移植为DCU代码，请更换为下列计时==========
  hipEventRecord(test_stop_event, 0);                                 // 禁止修改
  hipEventSynchronize(test_stop_event);                               // 禁止修改
  float test_time = 0.0;                                              // 禁止修改
  hipEventElapsedTime(&test_time, test_start_event, test_stop_event); // 禁止修改

  if (usparse::GMRES_OPT_VERIFY_SWITCH) {
    double ver_nrom = Verification(dev_A, dev_x, d_b, d_r0, N);
    if (ver_nrom < REL_RESID_LIMIT) {
      printf("verify precision : pass !   resid_ver =  %.14f  \n", ver_nrom);
    } else {
      printf("verify precision : no pass!  resid_ver = %.14f \n", ver_nrom);
    }
  }

  // destory data
  // todo: destroy csrmv info.
  destory_device_matrix(d_r0, d_V, d_s, d_H, d_cs, d_sn, d_vi);
  destory_device_float(d_V_f, d_H_f, d_vi_f, d_sn_f, d_cs_f, d_r0_f, d_A_val_f, d_s_f, d_b_f);

  HIP_CHECK(hipFree(d_beta_f));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_temp_buffer));
  return make_tuple(iteration, test_time, resid / init_res); // 禁止修改
}

// 此函数禁止修改
void initialize(SpM<double> *A, double *x, double *b) {
  int N = A->nrows;

  for (int i = 0; i < N; i++) {
    x[i] = sin(i);
  }

  double beta = calculateNorm(x, N);
  for (uint i = 0; i < N; i++) {
    x[i] /= beta;
  }

  spmv(A->rows, A->cols, A->vals, x, b, N);

  for (uint i = 0; i < N; i++)
    x[i] = 0.0;
}
