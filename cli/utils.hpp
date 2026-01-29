//
// Created by chu genshen on 2021/9/14.
//

#ifndef SPMV_ACC_UTILS_HPP
#define SPMV_ACC_UTILS_HPP

#include <iomanip>

#include "building_config.h"
#include "matrix_format/sparse_format.h"

#include "common/usparse_utils.hpp"

#define ROCSPARSE_CHECK(stat)                                                                                          \
  {                                                                                                                    \
    if ((stat) != rocsparse_status_success) {                                                                          \
      std::cerr << "Error: rocsparse error in line " << __LINE__ << std::endl;                                         \
      exit(-1);                                                                                                        \
    }                                                                                                                  \
  }

template <typename T> void print_vector(std::vector<T> &vecTest) {
  for (auto it : vecTest) {
    std::cout << std::setprecision(10) << it << " ";
  }
  std::cout << std::endl;
}

template <typename T> void print_vector(int n, T *x) {
  for (int i = 0; i < n; i++) {
    std::cout << x[i] << " ";
  }
  std::cout << std::endl;
  return;
}

inline double rand_double(double min, double max) {
  double temp = min + (max - min) * double(rand() % 100) / double((101));
  return temp;
}

template <typename T> void generate_vector(int n, T *&x) {
  x = new T[n];
  for (int i = 0; i < n; i++) {
    x[i] = static_cast<T>(rand_double(-1.0, 1.0));
  }
}

template <typename T> struct host_vectors {
  T *hX;
  T *temphY; // the origin y vector
  T *hY;     // store data copied from device side for later verification.
  T *hhY;    // store the correct results produced by host or device verification.
};

template <typename INDEX_TYPE, typename DATA_TYPE>
inline void create_host_data(coo_mtx<INDEX_TYPE, DATA_TYPE> _coo, host_vectors<DATA_TYPE> &h_vecs,
                             bool overwrite_hx = false) {
  DATA_TYPE *temp_coo_dense_vec;
  if (!overwrite_hx) {
    temp_coo_dense_vec = new DATA_TYPE[_coo.cols];
    memcpy(temp_coo_dense_vec, h_vecs.hX, _coo.cols * sizeof(DATA_TYPE)); // backup hX
  }

  generate_vector(_coo.cols, h_vecs.hX);
  generate_vector(_coo.rows, h_vecs.temphY);
  generate_vector(_coo.rows, h_vecs.hY); // 实际数据依然来自随机生成
  generate_vector(_coo.rows, h_vecs.hhY);

  // 统一主机端和设备端的向量值
  memcpy(h_vecs.hY, h_vecs.temphY, _coo.rows * sizeof(DATA_TYPE));
  memcpy(h_vecs.hhY, h_vecs.temphY, _coo.rows * sizeof(DATA_TYPE));

  if (!overwrite_hx) {
    memcpy(h_vecs.hX, temp_coo_dense_vec, _coo.cols * sizeof(DATA_TYPE)); // 外部读取数据恢复 hX
    delete[] temp_coo_dense_vec;
  }
}

template <typename INDEX_TYPE, typename DATA_TYPE>
inline void create_host_data(csr_mtx<INDEX_TYPE, DATA_TYPE> _csr, host_vectors<DATA_TYPE> &h_vecs,
                             bool overwrite_hx = false) {
  DATA_TYPE *temp_csr_dense_vec;
  if (!overwrite_hx) {
    temp_csr_dense_vec = new DATA_TYPE[_csr.cols];
    memcpy(temp_csr_dense_vec, h_vecs.hX, _csr.cols * sizeof(DATA_TYPE)); // backup hX
  }

  generate_vector(_csr.cols, h_vecs.hX);
  generate_vector(_csr.rows, h_vecs.temphY);
  generate_vector(_csr.rows, h_vecs.hY); // 实际数据依然来自随机生成
  generate_vector(_csr.rows, h_vecs.hhY);

  // 统一主机端和设备端的向量值
  memcpy(h_vecs.hY, h_vecs.temphY, _csr.rows * sizeof(DATA_TYPE));
  memcpy(h_vecs.hhY, h_vecs.temphY, _csr.rows * sizeof(DATA_TYPE));

  if (!overwrite_hx) {
    memcpy(h_vecs.hX, temp_csr_dense_vec, _csr.cols * sizeof(DATA_TYPE)); // 外部读取数据恢复 hX
    delete[] temp_csr_dense_vec;
  }
}

template <typename DATA_TYPE> inline void destroy_host_data(host_vectors<DATA_TYPE> &h_vecs) {
  delete[] h_vecs.hhY;
  delete[] h_vecs.hY;
  delete[] h_vecs.temphY;
  delete[] h_vecs.hX;
}

template <typename INDEX_TYPE, typename DATA_TYPE>
inline csr_mtx<INDEX_TYPE, DATA_TYPE> create_device_data(csr_mtx<INDEX_TYPE, DATA_TYPE> h_csr, DATA_TYPE *hX,
                                                         DATA_TYPE *hY, DATA_TYPE *&dX, DATA_TYPE *&dY) {
  const INDEX_TYPE A_num_rows = h_csr.rows;
  const INDEX_TYPE A_num_cols = h_csr.cols;
  const INDEX_TYPE A_nnz = h_csr.nnz;

  csr_mtx<INDEX_TYPE, DATA_TYPE> dev_csr;
  dev_csr.rows = A_num_rows;
  dev_csr.cols = A_num_cols;
  dev_csr.nnz = A_nnz;

  USPARSE_HIP_API_CHECK(hipMalloc((void **)&dev_csr.row_ptr, (A_num_rows + 1) * sizeof(INDEX_TYPE)))
  USPARSE_HIP_API_CHECK(hipMalloc((void **)&dev_csr.col_index, A_nnz * sizeof(INDEX_TYPE)))
  USPARSE_HIP_API_CHECK(hipMalloc((void **)&dev_csr.values, A_nnz * sizeof(DATA_TYPE)))
  USPARSE_HIP_API_CHECK(hipMalloc((void **)&dX, A_num_cols * sizeof(DATA_TYPE)))
  USPARSE_HIP_API_CHECK(hipMalloc((void **)&dY, A_num_rows * sizeof(DATA_TYPE)))

  USPARSE_HIP_API_CHECK(
      hipMemcpy(dev_csr.row_ptr, h_csr.row_ptr, (A_num_rows + 1) * sizeof(INDEX_TYPE), hipMemcpyHostToDevice))
  USPARSE_HIP_API_CHECK(
      hipMemcpy(dev_csr.col_index, h_csr.col_index, A_nnz * sizeof(INDEX_TYPE), hipMemcpyHostToDevice))
  USPARSE_HIP_API_CHECK(hipMemcpy(dev_csr.values, h_csr.values, A_nnz * sizeof(DATA_TYPE), hipMemcpyHostToDevice))
  USPARSE_HIP_API_CHECK(hipMemcpy(dX, hX, A_num_cols * sizeof(DATA_TYPE), hipMemcpyHostToDevice))
  USPARSE_HIP_API_CHECK(hipMemcpy(dY, hY, A_num_rows * sizeof(DATA_TYPE), hipMemcpyHostToDevice))
  return dev_csr;
}

template <typename INDEX_TYPE, typename DATA_TYPE>
inline coo_mtx<INDEX_TYPE, DATA_TYPE> create_device_data(coo_mtx<INDEX_TYPE, DATA_TYPE> h_coo, DATA_TYPE *hX,
                                                         DATA_TYPE *hY, DATA_TYPE *&dX, DATA_TYPE *&dY) {
  const INDEX_TYPE A_num_rows = h_coo.rows;
  const INDEX_TYPE A_num_cols = h_coo.cols;
  const INDEX_TYPE A_nnz = h_coo.nnz;

  coo_mtx<INDEX_TYPE, DATA_TYPE> dev_coo;
  dev_coo.rows = A_num_rows;
  dev_coo.cols = A_num_cols;
  dev_coo.nnz = A_nnz;

  USPARSE_HIP_API_CHECK(hipMalloc((void **)&dev_coo.row_index, A_nnz * sizeof(INDEX_TYPE)))
  USPARSE_HIP_API_CHECK(hipMalloc((void **)&dev_coo.col_index, A_nnz * sizeof(INDEX_TYPE)))
  USPARSE_HIP_API_CHECK(hipMalloc((void **)&dev_coo.values, A_nnz * sizeof(DATA_TYPE)))
  USPARSE_HIP_API_CHECK(hipMalloc((void **)&dX, A_num_cols * sizeof(DATA_TYPE)))
  USPARSE_HIP_API_CHECK(hipMalloc((void **)&dY, A_num_rows * sizeof(DATA_TYPE)))

  USPARSE_HIP_API_CHECK(
      hipMemcpy(dev_coo.row_index, h_coo.row_index, A_nnz * sizeof(INDEX_TYPE), hipMemcpyHostToDevice))
  USPARSE_HIP_API_CHECK(
      hipMemcpy(dev_coo.col_index, h_coo.col_index, A_nnz * sizeof(INDEX_TYPE), hipMemcpyHostToDevice))
  USPARSE_HIP_API_CHECK(hipMemcpy(dev_coo.values, h_coo.values, A_nnz * sizeof(DATA_TYPE), hipMemcpyHostToDevice))
  USPARSE_HIP_API_CHECK(hipMemcpy(dX, hX, A_num_cols * sizeof(DATA_TYPE), hipMemcpyHostToDevice))
  USPARSE_HIP_API_CHECK(hipMemcpy(dY, hY, A_num_rows * sizeof(DATA_TYPE), hipMemcpyHostToDevice))
  return dev_coo;
}

template <typename INDEX_TYPE, typename DATA_TYPE>
inline void destroy_device_data(csr_mtx<INDEX_TYPE, DATA_TYPE> d_csr, DATA_TYPE *&dX, DATA_TYPE *&dY) {
  USPARSE_HIP_API_CHECK(hipFree((void *)dY))
  USPARSE_HIP_API_CHECK(hipFree((void *)dX))
  USPARSE_HIP_API_CHECK(hipFree((void *)d_csr.values))
  USPARSE_HIP_API_CHECK(hipFree((void *)d_csr.col_index))
  USPARSE_HIP_API_CHECK(hipFree((void *)d_csr.row_ptr))
}

#endif // SPMV_ACC_UTILS_HPP
