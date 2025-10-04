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

template <typename T> usparse::usparse_csrmv_algo select_csrmv_algo(const csr_desc<int, T> d_csr_desc) {
  const float avg_nnz_per_row = static_cast<float>(d_csr_desc.nnz) / d_csr_desc.rows;
  const bool is_very_sparse = (avg_nnz_per_row < 7.0f);
  const bool is_medium_density = (avg_nnz_per_row >= 7.0f && avg_nnz_per_row < 25.0f);
  const bool is_very_large = (d_csr_desc.rows > 1000000);
  const bool is_medium_large = (d_csr_desc.rows >= 100000 && d_csr_desc.rows <= 300000);
  const bool is_low_medium_density = (avg_nnz_per_row >= 21.5f && avg_nnz_per_row < 25.0f);

  if ((is_very_sparse && is_medium_large) || (is_medium_density && is_very_large) ||
      (is_low_medium_density && is_medium_large)) {
    return usparse::usparse_csrmv_algo_line_enhance;
  }

  return usparse::usparse_csrmv_algo_flat;
}

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

// In this kernel, we:
// 1: set d_V <- d_b / beta.
// 2: set d_s <- zero.
// Note: we set d_s[0] <- beta using memory copy outside this kernel.
template <typename I, typename T>
__global__ void __launch_bounds__(512)
    big_fusion_outter_itl_v0(const T *d_b, T *d_V, T *d_s, const T beta, const I N, const I S_LEN) {
  const int gid = blockDim.x * blockIdx.x + threadIdx.x;
  const T beta_recip = 1 / beta;
  if (gid < N) {
    d_V[gid] = beta_recip * d_b[gid];
  }
  if (gid < S_LEN) {
    d_s[gid] = static_cast<T>(0);
  }
}

template <typename I, typename T>
void big_fusion_outter_itl_v0_wrapper(const T *d_b, T *d_V, T *d_s, const T beta, const I N, const I S_LEN) {
  constexpr int THREADS = 512;
  const int BLOCKS = N / THREADS + (N % THREADS == 0 ? 0 : 1);
  (big_fusion_outter_itl_v0<I, T>)<<<BLOCKS, THREADS>>>(d_b, d_V, d_s, beta, N, S_LEN);
}

template <typename I, typename T1, typename T2, int THREADS>
__global__ void __launch_bounds__(512) big_fusion_outter_itl_v1_part1(const T1 *dev_y, T2 *d_r0, const T2 *d_b, T2 *d_s,
                                                                      T2 *d_temp_buffer, const I N, const I S_LEN) {
  const int gid = blockDim.x * blockIdx.x + threadIdx.x;
  const int bid = blockIdx.x;
  const int tid_in_block = threadIdx.x;

  T2 local_sum = static_cast<T2>(0);
  if (gid < N) {
    local_sum = static_cast<T2>(dev_y[gid]) - d_b[gid];
    d_r0[gid] = local_sum;
    local_sum = local_sum * local_sum;
  }
  if (gid < S_LEN) {
    d_s[gid] = static_cast<T2>(0);
  }
  // reduce part1: reduce in each block
  const T2 block_sum =
      usparse::line_enhance_block_reduce<int, T2, THREADS, usparse::USPARSE_WAVEFRONT_SIZE>(local_sum, tid_in_block);
  if (tid_in_block == 0) {
    d_temp_buffer[blockIdx.x] = block_sum;
  }
}

template <typename I, typename T, int THREADS, bool IS_FINAL>
__global__ void __launch_bounds__(512)
    big_fusion_outter_itl_v1_part2(T *dev_result, T *d_temp_buffer, const I data_len) {
  const int tid_in_block = threadIdx.x;
  const int gid = blockDim.x * blockIdx.x + threadIdx.x;

  // reduce part2: reduce among blocks.
  T local_sum = static_cast<T>(0);

  if (IS_FINAL) { // data_len <= THREADS
    if (blockIdx.x == 0) {
      if (tid_in_block < data_len) {
        local_sum = d_temp_buffer[tid_in_block];
      }

      T final_sum =
          usparse::line_enhance_block_reduce<int, T, THREADS, usparse::USPARSE_WAVEFRONT_SIZE>(local_sum, tid_in_block);
      if (tid_in_block == 0) {
        dev_result[0] = sqrt(final_sum);
      }
    }
  } else { // data_len > THREADS
    T local_sum = static_cast<T>(0);
    if (gid < data_len) {
      local_sum = d_temp_buffer[gid];
    }
    const T block_sum =
        usparse::line_enhance_block_reduce<int, T, THREADS, usparse::USPARSE_WAVEFRONT_SIZE>(local_sum, tid_in_block);
    if (tid_in_block == 0) {
      d_temp_buffer[blockIdx.x] = block_sum;
    }
  }
}

template <typename I, typename T>
__global__ void __launch_bounds__(512)
    big_fusion_outter_itl_v1_part3(const T *d_r0, T *d_V, const T *d_beta, const I N) {
  const int gid = blockDim.x * blockIdx.x + threadIdx.x;
  const T beta = d_beta[0]; // bcast it.
  const T beta_recip = 1 / beta;
  if (gid < N) {
    d_V[gid] = -d_r0[gid] * beta_recip;
  }
}

template <typename I, typename T1, typename T2>
void big_fusion_outter_itl_v1(T1 *dev_y, T2 *d_r0, const T2 *d_b, T2 *d_V, T2 *d_s, T2 *d_temp_buffer, T2 *d_beta,
                              const I N, const I S_LEN) {
  constexpr int THREADS = 512;
  const int BLOCKS = N / THREADS + (N % THREADS == 0 ? 0 : 1);
  // convert data type and perform norm (reduce inside block)
  big_fusion_outter_itl_v1_part1<uint, double, float, THREADS>
      <<<BLOCKS, THREADS>>>(dev_y, d_r0, d_b, d_s, d_temp_buffer, N, S_LEN);
  // perform norm
  if (BLOCKS <= THREADS) {
    // reduce among `BLOCKS` blocks and sqrt
    big_fusion_outter_itl_v1_part2<uint, float, THREADS, true><<<1, THREADS>>>(d_beta, d_temp_buffer, BLOCKS);
  } else {
    const int SECOND_BLOCKS = BLOCKS / THREADS + (BLOCKS % THREADS == 0 ? 0 : 1);
    // reduce among `BLOCKS` blocks
    big_fusion_outter_itl_v1_part2<uint, float, THREADS, false>
        <<<SECOND_BLOCKS, THREADS>>>(d_beta, d_temp_buffer, BLOCKS);
    // reduce among `SECOND_BLOCKS` blocks and sqrt
    big_fusion_outter_itl_v1_part2<uint, float, THREADS, true><<<1, THREADS>>>(d_beta, d_temp_buffer, SECOND_BLOCKS);
  }
  // d_V = -d_r0/beta
  big_fusion_outter_itl_v1_part3<uint, float><<<BLOCKS, THREADS>>>(d_r0, d_V, d_beta, N);
}

template <typename I, typename ST, typename VT, typename XT, int II>
__global__ void big_fusion_v2_batched_axpy_kernel(const ST *d_s_f, const VT *d_V_f, XT *dev_x, const I N, const I VN) {
  // loop j in i+1:
  //   dev_x = dev_x + d_s_f[j] * d_V_f;
  // end loop
  const int gid = blockDim.x * blockIdx.x + threadIdx.x;
  if (gid < N) {
    VT x = static_cast<VT>(dev_x[gid]); // XT to VT
#pragma unroll
    for (int j = 0; j <= II; j++) {
      const ST alpha = d_s_f[j];
      x += alpha * d_V_f[VN * j + gid];
    }
    dev_x[gid] = static_cast<XT>(x);
  }
}

template <typename I, typename ST, typename VT, typename XT>
inline void big_fusion_v2_batched_axpy(const ST *d_s_f, const VT *d_V_f, XT *dev_x, const I ii, const I N, const I VN) {
  constexpr int THREADS = 512;
  const int BLOCKS = N / THREADS + (N % THREADS == 0 ? 0 : 1);

#define SWITCH_CASE_SHORT_CODE(II)                                                                                     \
  case (II):                                                                                                           \
    big_fusion_v2_batched_axpy_kernel<I, ST, VT, XT, II><<<BLOCKS, THREADS>>>(d_s_f, d_V_f, dev_x, N, VN);             \
    break;

  switch (ii) {
    SWITCH_CASE_SHORT_CODE(0);
    SWITCH_CASE_SHORT_CODE(1);
    SWITCH_CASE_SHORT_CODE(2);
    SWITCH_CASE_SHORT_CODE(3);
    SWITCH_CASE_SHORT_CODE(4);
    SWITCH_CASE_SHORT_CODE(5);
    SWITCH_CASE_SHORT_CODE(6);
    SWITCH_CASE_SHORT_CODE(7);
    SWITCH_CASE_SHORT_CODE(8);
    SWITCH_CASE_SHORT_CODE(9);
    SWITCH_CASE_SHORT_CODE(10);
    SWITCH_CASE_SHORT_CODE(11);
    SWITCH_CASE_SHORT_CODE(12);
    SWITCH_CASE_SHORT_CODE(13);
    SWITCH_CASE_SHORT_CODE(14);
    SWITCH_CASE_SHORT_CODE(15);
    SWITCH_CASE_SHORT_CODE(16);
    SWITCH_CASE_SHORT_CODE(17);
    SWITCH_CASE_SHORT_CODE(18);
    SWITCH_CASE_SHORT_CODE(19);
    SWITCH_CASE_SHORT_CODE(20);
  default:;
  }
#undef SWITCH_CASE_SHORT_CODE
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

  HIP_CHECK(hipMalloc(&d_b, N * sizeof(double)));
  HIP_CHECK(hipMemcpy(dev_x, x_d, N * sizeof(double), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b, _b, N * sizeof(double), hipMemcpyHostToDevice));

  utils::warmup();
  
  // auto start = std::chrono::high_resolution_clock::now(); // 禁止修改
  // =======如果已移植为DCU代码，请更换为下列计时==========
  hipEvent_t test_start_event, test_stop_event;
  sync_handle h_sync;
  hipEventCreate(&test_start_event);
  hipEventCreate(&test_stop_event);
  hipEventRecord(test_start_event, 0);
  // 任何对稀疏矩阵的预处理操作，如稀疏矩阵压缩格式转换、非零元数组或向量的精度转换、稀疏矩阵的非零元特征计算等，均需放在计时范围内，相关内存申请和释放除外

  if (usparse::GMRES_ORTHO_USE == usparse::GMRES_OPT_USE_GS) {
    HIP_CHECK(hipMalloc(&d_gs_buffer, RESTART_TIMES * (N / 256 + (N % 256 == 0 ? 0 : 1)) * sizeof(float)));
    // TODO: strategy selection.
  }

  // additional 1 float for d_beta_f.
  const int tmp_buf_len = N / 256 + (N % 256 == 0 ? 0 : 1);
  HIP_CHECK(hipMalloc(&d_temp_buffer, (tmp_buf_len + 1) * sizeof(float)));
  d_beta_f = d_temp_buffer + tmp_buf_len;

  // force conversion precision form dev_A.val to d_A_val_f
  usparse::device_cast(dev_A->vals, d_A_val_f, dev_A->nnz);
  usparse::device_cast(d_b, d_b_f, N);

  // initialize info descrA handle
  csr_desc<int, float> d_csr_desc_float(dev_A->nrows, dev_A->ncols, d_csr_desc.nnz, d_csr_desc.row_ptr,
                                        d_csr_desc.col_index, d_A_val_f);
  usparse::csrmv_mat_info csrmv_info;
  const usparse::usparse_csrmv_algo csrmv_algo = select_csrmv_algo(d_csr_desc);
  usparse::device_adaptive_analysis(0, 1.0, 0.0, d_csr_desc, csrmv_algo, &csrmv_info);


  const float f_density = static_cast<float>(dev_A->nnz) / (dev_A->nrows * dev_A->ncols);

  // 任何对稀疏矩阵的预处理操作，如稀疏矩阵压缩格式转换、非零元数组或向量的精度转换、稀疏矩阵的非零元特征计算等，均需放在计时范围内，相关内存申请和释放除外

  /****GMRES计算过程****/
  do {
    // ==========外迭代============
    // spmv(A_d->rows, A_d->cols, A_d->vals, x_d, r0.data(), N); // 不可修改此步操作中相关数据的存储精度和SpMV计算精度
    // copy x from host to device

    const uint S_LEN = RESTART_TIMES + 1; // the length of array s. (we assume S_LEN < N).
    constexpr bool USE_FUSION = true;     // todo: move to config file.

    // if it is the first time, dev_x is 0. Ax is also 0.
    if (iteration != 0) {
      // compute y = Ax
      usparse::device_adaptive_spmv<int, double, BEAT_Y_POLICY_ZERO>(0, 1.0, 0.0, d_csr_desc, dev_x, dev_y, csrmv_algo,
                                                                     &csrmv_info);
      if (!USE_FUSION) {
        usparse::device_cast<uint, double, float>(dev_y, d_r0_f, N); // dev_y => d_r0_f

        // float
        usparse::device_daxpy<uint, float, float>(-1.0, d_b_f, d_r0_f, N); // d_r0_f += -1.0 * d_b_f.

        usparse::device_calculateNorm<uint, float>(d_r0_f, N, d_beta_f, d_temp_buffer, 0);
        h_sync.sync();

        beta = *d_beta_f;
        usparse::device_dscal<uint, float>(-1.0 / beta, d_r0_f, N);

        usparse::device_dcopy<uint, float>(d_r0_f, d_V_f, N);

        usparse::device_memset<uint, float>(d_s_f, (RESTART_TIMES + 1));
        HIP_CHECK(hipMemcpy(&d_s_f[0], d_beta_f, sizeof(float), hipMemcpyDeviceToDevice));
      } else {
        // input: dev_y (double[])
        // output: d_r0_f <- (float [])(dev_y) - d_b_f
        // output: beta: norm(d_r0_f)
        // output: d_s_f <- 0.0;
        // output: d_beta_f[0]: beta

        // output: d_V_f <- -d_r0_f/beta
        // output: d_s_f[0] = beta.

        big_fusion_outter_itl_v1<uint, double, float>(dev_y, d_r0_f, d_b_f, d_V_f, d_s_f, d_temp_buffer, d_beta_f, N,
                                                      S_LEN);
        float beta_host_f32 = 0.0;
        HIP_CHECK(hipMemcpy(&beta_host_f32, d_beta_f, sizeof(float), hipMemcpyDeviceToHost)); // sync kernel
        beta = beta_host_f32;                                                                 // fp32 to fp64

        // note: d2d can be async on DCU, but we dont need sync. Because it is ready before using `d_s_f`.
        HIP_CHECK(hipMemcpy(&d_s_f[0], d_beta_f, sizeof(float), hipMemcpyDeviceToDevice));
      }
    } else {
      // if it is the first iteration. the above code in if branch can be simplified as following.
      // d_V_f = d_r0_f = -d_b_f.
      if (!USE_FUSION) {
        usparse::device_copy_v2(d_b_f, d_r0_f, (float)(-1.0), N);
        // d_beta_f = norm(-d_b_f); which is also beta.
        // d_s_f[0] = d_beta_f[0] = (float)beta.
        *d_beta_f = (float)beta;

        usparse::device_dscal<uint, float>(-1.0 / beta, d_r0_f, N);

        usparse::device_dcopy<uint, float>(d_r0_f, d_V_f, N);

        usparse::device_memset<uint, float>(d_s_f, (RESTART_TIMES + 1));
        HIP_CHECK(hipMemcpy(&d_s_f[0], d_beta_f, sizeof(float), hipMemcpyDeviceToDevice));
      } else {
        // input: d_b_f
        // output: d_r0_f: (d_b_f) * (1.0/beta). (omitted. because d_r0_f is overwrite later)
        // output: d_V_f: d_r0_f
        // output: d_s_f: 0.0; d_s_f[0] = beta.
        // output: d_beta_f: beta. (omitted)
        big_fusion_outter_itl_v0_wrapper<uint, float>(d_b_f, d_V_f, d_s_f, beta, N, S_LEN);
        const float beta_fp32 = beta;
        // copying d_s_f does not need sync: it is ready before using `d_s_f`.
        HIP_CHECK(hipMemcpy(&d_s_f[0], &beta_fp32, sizeof(float), hipMemcpyHostToDevice));
      }
    }

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

      int h_idx = (i + 3) * i / 2;
      if (usparse::GMRES_ORTHO_USE == usparse::GMRES_OPT_USE_MGS) {
        if (usparse::GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION ==
            usparse::GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION_LEVEL_2) {
          usparse::dotprod_axpy_fusion<uint, float, usparse::GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION_LEVEL_2>(
              d_r0_f, d_V_f, d_H_f, d_temp_buffer, N, RESTART_TIMES, i, 0, f_density);
        } else if (usparse::GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION ==
                   usparse::GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION_LEVEL_1) {
          usparse::dotprod_axpy_fusion<uint, float, usparse::GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION_LEVEL_1>(
              d_r0_f, d_V_f, d_H_f, d_temp_buffer, N, RESTART_TIMES, i, 0, f_density);
          usparse::device_calculateNorm(d_r0_f, N, &d_H_f[h_idx + i + 1], d_temp_buffer, 0);
          h_sync.sync();
          alpha = 1.0 / d_H_f[h_idx + i + 1];
          usparse::device_dscal<uint, float>(alpha, d_r0_f, N);
          usparse::device_dcopy(d_r0_f, d_V_f + N * (i + 1), N);
        } else { // GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION_NONE
          for (k = 0; k <= i; k++) {
            usparse::device_dotProduct(d_r0_f, d_V_f + k * N, N, &d_H_f[h_idx + k], d_temp_buffer, 0);
            h_sync.sync();
            alpha = -d_H_f[h_idx + k];
            usparse::device_daxpy<uint, float, float>(alpha, d_V_f + N * k, d_r0_f, N);
          }
          usparse::device_calculateNorm(d_r0_f, N, &d_H_f[h_idx + i + 1], d_temp_buffer, 0);
          h_sync.sync();

          alpha = 1.0 / d_H_f[(h_idx + i + 1)];
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
            usparse::device_dotProduct(d_r0_f, d_V_f + k * N, N, &d_H_f[h_idx + k], d_temp_buffer, 0);
          }
          h_sync.sync();
          for (k = 0; k <= i; k++) {
            alpha = -d_H_f[h_idx + k];
            usparse::device_daxpy<uint, float, float>(alpha, d_V_f + N * k, d_r0_f, N);
          }
          usparse::device_calculateNorm(d_r0_f, N, &d_H_f[h_idx + (i + 1)], d_temp_buffer, 0);
          h_sync.sync();

          alpha = 1.0 / d_H_f[h_idx + (i + 1)];
          usparse::device_dscal<uint, float>(alpha, d_r0_f, N);
          usparse::device_dcopy(d_r0_f, d_V_f + N * (i + 1), N);
        }
      }

      float tmp_fp32;
      if (usparse::GMRES_USE_HOST_ROTATION) {
        h_sync.sync();
        usparse::host_rotation2((uint)(RESTART_TIMES), d_H_f, d_cs_f, d_sn_f, d_s_f, (uint)(i)); // todo: uint to int
        tmp = d_s_f[i + 1];
      } else {
        usparse::device_rotation2((uint)(RESTART_TIMES), d_H_f + h_idx, d_cs_f, d_sn_f, d_s_f,
                                  (uint)(i)); // todo: uint to int
        // h_sync.sync();
        hipMemcpy(&tmp_fp32, d_s_f + i + 1, sizeof(float), hipMemcpyDeviceToHost);
        tmp = tmp_fp32;
      }

      resid = std::abs(tmp);
      // std::cout << "iteration " << iteration << ", resid = " << resid/init_res << std::endl;

      if (resid <= RESID_LIMIT || iteration >= ITERATION_LIMIT) {
        break;
      }
    } while (i + 1 < RESTART_TIMES && iteration <= ITERATION_LIMIT);

    constexpr bool USE_BATCHED_AXPY = USE_FUSION;
    // 求解上三角系统
    if (usparse::GMRES_USE_HOST_TRI_SOLVER) {
      h_sync.sync(); // wait kernel to finish
      usparse::host_sovlerTri(RESTART_TIMES, i, d_H_f, d_s_f);
    } else {
      usparse::device_sovlerTri(RESTART_TIMES, i, d_H_f, d_s_f);
      if (!USE_BATCHED_AXPY) {
        h_sync.sync();
      }
    }

    // 更新解: <float, double>
    if (!USE_BATCHED_AXPY) { // naive approach
      for (j = 0; j <= i; j++) {
        const float tmp = d_s_f[j];
        usparse::device_daxpy<uint, float, double>(tmp, d_V_f + j * N, dev_x, N);
      }
    } else { // batched axpy
      if (i == 0) {
        for (j = 0; j <= i; j++) {
          h_sync.sync();
          tmp = d_s_f[j]; // todo: use memory load.
          usparse::device_daxpy<uint, float, double>(tmp, d_V_f + j * N, dev_x, N);
        }
      } else {
        // todo: change N+1 if V length channges
        big_fusion_v2_batched_axpy<uint, float, float, double>(d_s_f, d_V_f, dev_x, i, N, N + 0);
      }
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

  // HIP_CHECK(hipFree(d_beta_f));
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
