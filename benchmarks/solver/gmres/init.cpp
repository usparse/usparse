
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "init.hpp"
#include "kernels/gmres_optimization_config.h"
#include "kernels/usparse_gmres.h"

usparse::var_csr_desc<int, double> init_device_data(usparse::csr_mat_descr<double, int> h_A, double *h_x,
                                                    double **dev_x, double **dev_y) {
  const int A_num_rows = h_A.rows;
  const int A_num_cols = h_A.cols;
  const int A_nnz = h_A.nnz;

  usparse::var_csr_desc<int, double> dev_A;
  // device A
  dev_A.rows = A_num_rows;
  dev_A.cols = A_num_cols;
  dev_A.nnz = A_nnz;

  // malloc A x y on device
  USPARSE_HIP_API_CHECK(hipMalloc((void **)&(dev_A.row_ptr), (A_num_rows + 1) * sizeof(int)));
  USPARSE_HIP_API_CHECK(hipMalloc((void **)&(dev_A.col_index), A_nnz * sizeof(int)));
  USPARSE_HIP_API_CHECK(hipMalloc((void **)&(dev_A.values), A_nnz * sizeof(double)));
  USPARSE_HIP_API_CHECK(hipMalloc((void **)dev_x, A_num_rows * sizeof(double)));
  USPARSE_HIP_API_CHECK(hipMalloc((void **)dev_y, A_num_rows * sizeof(double)));

  // cpoy A from host to device
  USPARSE_HIP_API_CHECK(hipMemcpy(dev_A.row_ptr, h_A.row_ptr, (A_num_rows + 1) * sizeof(int), hipMemcpyHostToDevice));
  USPARSE_HIP_API_CHECK(hipMemcpy(dev_A.col_index, h_A.col_index, A_nnz * sizeof(int), hipMemcpyHostToDevice));
  USPARSE_HIP_API_CHECK(hipMemcpy(dev_A.values, h_A.values, A_nnz * sizeof(double), hipMemcpyHostToDevice));
  USPARSE_HIP_API_CHECK(hipMemcpy(*dev_x, h_x, A_num_rows * sizeof(double), hipMemcpyHostToDevice));
  return dev_A;
}

/**
 * initialize matrix info
 */
void init_matrix_info(const usparse::csr_mat_descr<double, int> dev_A, rocsparse_handle *handle,
                      rocsparse_mat_descr *descrA, rocsparse_mat_info *info) {
#ifdef __USPARSE_BUILD_BACKEND_ROCM__
  rocsparse_create_handle(handle);
  // Matrix descriptor
  // rocsparse_mat_descr descrA;
  rocsparse_create_mat_descr(descrA);
  // Create meta data
  // rocsparse_mat_info info;
  rocsparse_create_mat_info(info);
  // Analyse CSR matrix
  rocsparse_dcsrmv_analysis(*handle, rocsparse_operation_none, dev_A.rows, dev_A.cols, dev_A.nnz, *descrA, dev_A.values,
                            dev_A.row_ptr, dev_A.col_index, *info);
#endif
}

void destory_device_data(usparse::var_csr_desc<int, double> dev_A, double *d_y, double *d_x) {
  USPARSE_HIP_API_CHECK(hipFree(dev_A.row_ptr));
  USPARSE_HIP_API_CHECK(hipFree(dev_A.col_index));
  USPARSE_HIP_API_CHECK(hipFree(dev_A.values));
  USPARSE_HIP_API_CHECK(hipFree(d_y));
  USPARSE_HIP_API_CHECK(hipFree(d_x));
}

void destory_device_matrix(double *d_r0, double *d_V, double *d_s, double *d_H, double *d_cs, double *d_sn,
                           double *d_vi) {
  USPARSE_HIP_API_CHECK(hipFree(d_r0));
  USPARSE_HIP_API_CHECK(hipFree(d_V));
  USPARSE_HIP_API_CHECK(hipFree(d_s));
  USPARSE_HIP_API_CHECK(hipFree(d_H));
  USPARSE_HIP_API_CHECK(hipFree(d_cs));
  USPARSE_HIP_API_CHECK(hipFree(d_sn));
  USPARSE_HIP_API_CHECK(hipFree(d_vi));
}

void destory_descr_info(rocsparse_mat_info info, rocsparse_handle handle, rocsparse_mat_descr descrA) {
#ifdef __USPARSE_BUILD_BACKEND_ROCM__
  rocsparse_destroy_mat_info(info);
  rocsparse_destroy_mat_descr(descrA);
  rocsparse_destroy_handle(handle);
#endif
}

void init_device_matrix(double **d_r0, double **d_V, double **d_s, double **d_H, double **d_cs, double **d_sn,
                        double **d_vi, uint N, int m) {
  // init matrix
  USPARSE_HIP_API_CHECK(hipMalloc((void **)d_r0, N * sizeof(double)));
  USPARSE_HIP_API_CHECK(hipMalloc((void **)d_V, (m + 1) * N * sizeof(double)));
  USPARSE_HIP_API_CHECK(hipMalloc((void **)d_s, (m + 1) * sizeof(double)));
  USPARSE_HIP_API_CHECK(hipMalloc((void **)d_H, (m + 1) * m * sizeof(double)));
  USPARSE_HIP_API_CHECK(hipMalloc((void **)d_cs, m * sizeof(double)));
  USPARSE_HIP_API_CHECK(hipMalloc((void **)d_sn, m * sizeof(double)));
  USPARSE_HIP_API_CHECK(hipMalloc((void **)d_vi, N * sizeof(double)));

  // s设置为0.0
  USPARSE_HIP_API_CHECK(hipMemset(*d_s, 0.0, (m + 1) * sizeof(double)));
  USPARSE_HIP_API_CHECK(hipMemset(*d_r0, 0.0, N * sizeof(double)));
}

void init_device_matrix_float(float **d_V_f, float **d_H_f, float **d_vi_f, float **d_sn_f, float **d_cs_f,
                              float **d_r0_f, float **d_A_val_f, float **d_s_f, float **d_b_f, uint N, int m,
                              uint nnz) {
  const int VN = usparse::get_r0_v_mem_len(N);
  // init matrix
  USPARSE_HIP_API_CHECK(hipMalloc((void **)d_V_f, (m + 1) * VN * sizeof(float)));
  USPARSE_HIP_API_CHECK(hipMemset(*d_V_f, 0, (m + 1) * VN * sizeof(float)));
  USPARSE_HIP_API_CHECK(hipMalloc((void **)d_H_f, ((m + 3) * m ) / 2 * sizeof(float)));
  USPARSE_HIP_API_CHECK(hipMalloc((void **)d_vi_f, N * sizeof(float)));
  USPARSE_HIP_API_CHECK(hipMalloc((void **)d_cs_f, m * sizeof(float)));
  USPARSE_HIP_API_CHECK(hipMalloc((void **)d_sn_f, m * sizeof(float)));
  USPARSE_HIP_API_CHECK(hipMalloc((void **)d_r0_f, VN * sizeof(float)));
  USPARSE_HIP_API_CHECK(hipMemset(*d_r0_f, 0, VN * sizeof(float)));
  USPARSE_HIP_API_CHECK(hipMalloc((void **)d_A_val_f, nnz * sizeof(float)));
  USPARSE_HIP_API_CHECK(hipMalloc((void **)d_s_f, (m + 1) * sizeof(float)));
  USPARSE_HIP_API_CHECK(hipMalloc((void **)d_b_f, N * sizeof(float)));
}

void destory_device_float(float *d_V_f, float *d_H_f, float *d_vi_f, float *d_sn_f, float *d_cs_f, float *d_r0_f,
                          float *d_A_val_f, float *d_s_f, float *d_b_f) {
  USPARSE_HIP_API_CHECK(hipFree(d_r0_f));
  USPARSE_HIP_API_CHECK(hipFree(d_V_f));
  USPARSE_HIP_API_CHECK(hipFree(d_A_val_f));
  USPARSE_HIP_API_CHECK(hipFree(d_H_f));
  USPARSE_HIP_API_CHECK(hipFree(d_cs_f));
  USPARSE_HIP_API_CHECK(hipFree(d_sn_f));
  USPARSE_HIP_API_CHECK(hipFree(d_vi_f));
  USPARSE_HIP_API_CHECK(hipFree(d_s_f));
  USPARSE_HIP_API_CHECK(hipFree(d_b_f));
}

void initDCU(int device_id) {
  int device_count = 0;
  USPARSE_HIP_API_CHECK(hipGetDeviceCount(&device_count));
  std::cerr << "##  count == " << device_count << std::endl;
  if (device_count == 0) {
    std::cerr << "No HIP devices found!" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (device_id >= device_count) {
    std::cerr << "Invalid device ID " << device_id << ", only " << device_count << " devices available." << std::endl;
    exit(EXIT_FAILURE);
  }

  USPARSE_HIP_API_CHECK(hipSetDevice(device_id));

  hipDeviceProp_t props;
  USPARSE_HIP_API_CHECK(hipGetDeviceProperties(&props, device_id));
  std::cout << "Using device " << device_id << ": " << props.name << std::endl;
}

template <typename I, typename T>
void spmv(const I *rowPtr, const I *colInd, const T *values, const T *x, T *y, I numRows) {
  for (I i = 0; i < numRows; ++i) {
    T sum = 0.0;
    for (I j = rowPtr[i]; j < rowPtr[i + 1]; ++j) {
      sum += values[j] * x[colInd[j]];
    }
    y[i] = sum;
  }
}

double calculateNorm(const double *vec, uint N);

void initialize(const usparse::csr_mat_descr<double, int> A, double *x, double *b) {
  const int N = A.rows;

  for (int i = 0; i < N; i++) {
    x[i] = sin(i);
  }

  double beta = calculateNorm(x, N);
  for (uint i = 0; i < N; i++) {
    x[i] /= beta;
  }

  spmv(A.row_ptr, A.col_index, A.values, x, b, N);

  for (uint i = 0; i < N; i++)
    x[i] = 0.0;
}
