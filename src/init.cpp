#include "init.hpp"
#include <sys/types.h>

void init_device_data(SpM<double> *h_A, double *h_x, SpM<double> *dev_A, double **dev_x, double **dev_y) {
  //
  const int A_num_rows = h_A->nrows;
  const int A_num_cols = h_A->ncols;
  const int A_nnz = h_A->nnz;

  // device A
  dev_A->nrows = A_num_rows;
  dev_A->ncols = A_num_cols;
  dev_A->nnz = A_nnz;

  // malloc A x y on device
  HIP_CHECK(hipMalloc((void **)&dev_A->rows, (A_num_rows + 1) * sizeof(int)));
  HIP_CHECK(hipMalloc((void **)&dev_A->cols, A_nnz * sizeof(int)));
  HIP_CHECK(hipMalloc((void **)&dev_A->vals, A_nnz * sizeof(double)));
  HIP_CHECK(hipMalloc((void **)dev_x, A_num_rows * sizeof(double)));
  HIP_CHECK(hipMalloc((void **)dev_y, A_num_rows * sizeof(double)));

  // cpoy A from host to device
  HIP_CHECK(hipMemcpy(dev_A->rows, h_A->rows, (A_num_rows + 1) * sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dev_A->cols, h_A->cols, A_nnz * sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dev_A->vals, h_A->vals, A_nnz * sizeof(double), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(*dev_x, h_x, A_num_rows * sizeof(double), hipMemcpyHostToDevice));

  dev_A->on_device = true;
}

/**
 * initialize matrix info
 */
void init_matrix_info(SpM<double> *dev_A, rocsparse_handle *handle, rocsparse_mat_descr *descrA,
                      rocsparse_mat_info *info) {
  rocsparse_create_handle(handle);
  // Matrix descriptor
  // rocsparse_mat_descr descrA;
  rocsparse_create_mat_descr(descrA);
  // Create meta data
  // rocsparse_mat_info info;
  rocsparse_create_mat_info(info);
  // Analyse CSR matrix
  rocsparse_dcsrmv_analysis(*handle, rocsparse_operation_none, dev_A->nrows, dev_A->ncols, dev_A->nnz, *descrA,
                            dev_A->vals, (int *)dev_A->rows, (int *)dev_A->cols, *info);
}

void destory_device_data(SpM<double> *dev_A, double *d_y, double *d_x) {
  HIP_CHECK(hipFree(dev_A->rows));
  HIP_CHECK(hipFree(dev_A->cols));
  HIP_CHECK(hipFree(dev_A->vals));
  HIP_CHECK(hipFree(d_y));
  HIP_CHECK(hipFree(d_x));
}

void destory_device_matrix(double *d_r0, double *d_V, double *d_s, double *d_H, double *d_cs, double *d_sn,
                           double *d_vi) {
  HIP_CHECK(hipFree(d_r0));
  HIP_CHECK(hipFree(d_V));
  HIP_CHECK(hipFree(d_s));
  HIP_CHECK(hipFree(d_H));
  HIP_CHECK(hipFree(d_cs));
  HIP_CHECK(hipFree(d_sn));
  HIP_CHECK(hipFree(d_vi));
}

void destory_descr_info(rocsparse_mat_info info, rocsparse_handle handle, rocsparse_mat_descr descrA) {
  rocsparse_destroy_mat_info(info);
  rocsparse_destroy_mat_descr(descrA);
  rocsparse_destroy_handle(handle);
}

void init_device_matrix(double **d_r0, double **d_V, double **d_s, double **d_H, double **d_cs, double **d_sn,
                        double **d_vi, uint N, int m) {
  // init matrix
  HIP_CHECK(hipMalloc((void **)d_r0, N * sizeof(double)));
  HIP_CHECK(hipMalloc((void **)d_V, (m + 1) * N * sizeof(double)));
  HIP_CHECK(hipMalloc((void **)d_s, (m + 1) * sizeof(double)));
  HIP_CHECK(hipMalloc((void **)d_H, (m + 1) * m * sizeof(double)));
  HIP_CHECK(hipMalloc((void **)d_cs, m * sizeof(double)));
  HIP_CHECK(hipMalloc((void **)d_sn, m * sizeof(double)));
  HIP_CHECK(hipMalloc((void **)d_vi, N * sizeof(double)));

  // s设置为0.0
  HIP_CHECK(hipMemset(*d_s, 0.0, (m + 1) * sizeof(double)));
}

void initDCU(int device_id) {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  std::cerr << "##  count == " << device_count << std::endl;
  if (device_count == 0) {
    std::cerr << "No HIP devices found!" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (device_id >= device_count) {
    std::cerr << "Invalid device ID " << device_id << ", only " << device_count << " devices available." << std::endl;
    exit(EXIT_FAILURE);
  }

  HIP_CHECK(hipSetDevice(device_id));

  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, device_id));
  std::cout << "Using device " << device_id << ": " << props.name << std::endl;
}
