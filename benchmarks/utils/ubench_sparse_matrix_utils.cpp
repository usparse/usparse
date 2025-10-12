//
// Created by genshen on 2025/10/09.
//
#include <iostream>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "include/usparse.h"

#include "ubench_utils.h"
#include "utils.hpp"

namespace ubench {
  /**
   * malloc device space for CSR matrix
   * @tparam I index date type
   * @tparam T date type for matrix values.
   *
   * @param dev_csr matrix descriptor for CSR matrix on device side
   * @param rows rows of the matrix
   * @param cols column of the matrix
   * @param nnz number of non-zeros of the CSR matrix
   */
  template <typename I, typename T>
  void ubench_malloc_csr_device(usparse::var_csr_desc<I, T> *dev_csr, const I rows, const I cols, const I nnz) {
    HIP_CHECK(hipMalloc((void **)(&(dev_csr->row_ptr)), (rows + 1) * sizeof(I)))
    HIP_CHECK(hipMalloc((void **)(&(dev_csr->col_index)), nnz * sizeof(I)))
    HIP_CHECK(hipMalloc((void **)(&(dev_csr->values)), nnz * sizeof(T)))
  }

  /**
   * copy data from host space to device space for CSR matrix
   * @tparam I index date type
   * @tparam T date type for matrix values.
   *
   * @param host_csr matrix descriptor for CSR matrix on host side (source data)
   * @param dev_csr matrix descriptor for CSR matrix on device side (destination data)
   */
  template <typename I, typename T>
  void ubench_copy_csr_H2D(const usparse::csr_desc<I, T> host_csr, usparse::var_csr_desc<I, T> *dev_csr) {
    HIP_CHECK(hipMemcpy(dev_csr->row_ptr, host_csr.row_ptr, (host_csr.rows + 1) * sizeof(I), hipMemcpyHostToDevice))
    HIP_CHECK(hipMemcpy(dev_csr->col_index, host_csr.col_index, host_csr.nnz * sizeof(I), hipMemcpyHostToDevice))
    HIP_CHECK(hipMemcpy(dev_csr->values, host_csr.values, host_csr.nnz * sizeof(T), hipMemcpyHostToDevice))
  }

  /**
   * free data space of CSR matrix malloced by `ubench_malloc_csr_device`
   * @tparam I index date type
   * @tparam T date type for matrix values.
   *
   * @param dev_csr matrix descriptor for CSR matrix on device side.
   */
  template <typename I, typename T> void ubench_free_device_csr(usparse::var_csr_desc<I, T> dev_csr) {
    HIP_CHECK(hipFree((void *)dev_csr.values))
    HIP_CHECK(hipFree((void *)dev_csr.col_index))
    HIP_CHECK(hipFree((void *)dev_csr.row_ptr))
  }

  // template instance for data manager of CSR matrix
  template void ubench_malloc_csr_device<usparse::usparse_int, usparse::usparse_fp32>(
      usparse::var_csr_desc<usparse::usparse_int, usparse::usparse_fp32> *dev_csr, const usparse::usparse_int rows,
      const usparse::usparse_int cols, const usparse::usparse_int nnz);

  template void ubench_malloc_csr_device<usparse::usparse_int, usparse::usparse_fp64>(
      usparse::var_csr_desc<usparse::usparse_int, usparse::usparse_fp64> *dev_csr, const usparse::usparse_int rows,
      const usparse::usparse_int cols, const usparse::usparse_int nnz);

  template void ubench_copy_csr_H2D<usparse::usparse_int, usparse::usparse_fp32>(
      const usparse::csr_desc<usparse::usparse_int, usparse::usparse_fp32> host_csr,
      usparse::var_csr_desc<usparse::usparse_int, usparse::usparse_fp32> *dev_csr);

  template void ubench_copy_csr_H2D<usparse::usparse_int, usparse::usparse_fp64>(
      const usparse::csr_desc<usparse::usparse_int, usparse::usparse_fp64> host_csr,
      usparse::var_csr_desc<usparse::usparse_int, usparse::usparse_fp64> *dev_csr);

  template void ubench_free_device_csr<usparse::usparse_int, usparse::usparse_fp32>(
      usparse::var_csr_desc<usparse::usparse_int, usparse::usparse_fp32> dev_csr);

  template void ubench_free_device_csr<usparse::usparse_int, usparse::usparse_fp64>(
      usparse::var_csr_desc<usparse::usparse_int, usparse::usparse_fp64> dev_csr);

  /**
   * malloc device space for COO matrix
   * @tparam I index date type
   * @tparam T date type for matrix values.
   *
   * @param dev_coo matrix descriptor for COO matrix on device side
   * @param rows rows of the matrix
   * @param cols column of the matrix
   * @param nnz number of non-zeros of the COO matrix
   */
  template <typename I, typename T>
  void ubench_malloc_coo_device(usparse::var_coo_desc<I, T> *dev_coo, const I rows, const I cols, const I nnz) {
    HIP_CHECK(hipMalloc((void **)(&(dev_coo->row_index)), nnz * sizeof(I)))
    HIP_CHECK(hipMalloc((void **)(&(dev_coo->col_index)), nnz * sizeof(I)))
    HIP_CHECK(hipMalloc((void **)(&(dev_coo->values)), nnz * sizeof(T)))
  }

  /**
   * malloc data from host space to device space for COO matrix
   * @tparam I index date type
   * @tparam T date type for matrix values.
   *
   * @param host_coo matrix descriptor for COO matrix on host side (source data)
   * @param dev_coo matrix descriptor for COO matrix on device side (destination data)
   */
  template <typename I, typename T>
  void ubench_copy_coo_H2D(const usparse::coo_desc<I, T> host_coo, usparse::var_coo_desc<I, T> *dev_coo) {
    HIP_CHECK(hipMemcpy(dev_coo->row_index, host_coo.row_index, host_coo.nnz * sizeof(I), hipMemcpyHostToDevice))
    HIP_CHECK(hipMemcpy(dev_coo->col_index, host_coo.col_index, host_coo.nnz * sizeof(I), hipMemcpyHostToDevice))
    HIP_CHECK(hipMemcpy(dev_coo->values, host_coo.values, host_coo.nnz * sizeof(T), hipMemcpyHostToDevice))
  }

  /**
   * free data space of COO matrix malloced by `ubench_malloc_coo_device`
   * @tparam I index date type
   * @tparam T date type for matrix values.
   *
   * @param dev_coo matrix descriptor for COO matrix on device side.
   */
  template <typename I, typename T> void ubench_free_device_coo(usparse::var_coo_desc<I, T> dev_coo) {
    HIP_CHECK(hipFree((void *)dev_coo.row_index))
    HIP_CHECK(hipFree((void *)dev_coo.col_index))
    HIP_CHECK(hipFree((void *)dev_coo.values))
  }

  // template instance for data manager of COO matrix
  template void ubench_malloc_coo_device<usparse::usparse_int, usparse::usparse_fp32>(
      usparse::var_coo_desc<usparse::usparse_int, usparse::usparse_fp32> *dev_coo, const usparse::usparse_int rows,
      const usparse::usparse_int cols, const usparse::usparse_int nnz);

  template void ubench_malloc_coo_device<usparse::usparse_int, usparse::usparse_fp64>(
      usparse::var_coo_desc<usparse::usparse_int, usparse::usparse_fp64> *dev_coo, const usparse::usparse_int rows,
      const usparse::usparse_int cols, const usparse::usparse_int nnz);

  template void ubench_copy_coo_H2D<usparse::usparse_int, usparse::usparse_fp32>(
      const usparse::coo_desc<usparse::usparse_int, usparse::usparse_fp32> host_coo,
      usparse::var_coo_desc<usparse::usparse_int, usparse::usparse_fp32> *dev_coo);

  template void ubench_copy_coo_H2D<usparse::usparse_int, usparse::usparse_fp64>(
      const usparse::coo_desc<usparse::usparse_int, usparse::usparse_fp64> host_coo,
      usparse::var_coo_desc<usparse::usparse_int, usparse::usparse_fp64> *dev_coo);

  template void ubench_free_device_coo<usparse::usparse_int, usparse::usparse_fp32>(
      usparse::var_coo_desc<usparse::usparse_int, usparse::usparse_fp32> dev_coo);

  template void ubench_free_device_coo<usparse::usparse_int, usparse::usparse_fp64>(
      usparse::var_coo_desc<usparse::usparse_int, usparse::usparse_fp64> dev_coo);
} // namespace ubench
