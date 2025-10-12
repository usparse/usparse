//
// Created by chu genshen on 2025/10/09.
//

#ifndef USPARSE_BENCHMARK_UTILS_H
#define USPARSE_BENCHMARK_UTILS_H

#include "include/usparse.h"

namespace ubench {
  template <typename I, typename T>
  void ubench_malloc_csr_device(usparse::var_csr_desc<I, T> *dev_csr, const I rows, const I cols, const I nnz);

  template <typename I, typename T>
  void ubench_copy_csr_H2D(const usparse::csr_desc<I, T> host_csr, usparse::var_csr_desc<I, T> *dev_csr);

  template <typename I, typename T> void ubench_free_device_csr(usparse::var_csr_desc<I, T> dev_csr);

  template <typename I, typename T>
  void ubench_malloc_coo_device(usparse::var_coo_desc<I, T> *dev_coo, const I rows, const I cols, const I nnz);

  template <typename I, typename T>
  void ubench_copy_coo_H2D(const usparse::coo_desc<I, T> host_coo, usparse::var_coo_desc<I, T> *dev_coo);

  template <typename I, typename T> void ubench_free_device_coo(usparse::var_coo_desc<I, T> dev_coo);

  template <typename I, typename T> void generate_vector(T *x, int N);

  template <typename I, typename T, bool MALLOC_HOST = false>
  void ubench_malloc_and_init_host_dev_vector(T **host_vec, T **dev_vec, I N);

  template <typename I, typename T, bool FREE_HOST = false>
  void ubench_free_host_dev_vector(T *host_vec, T *dev_vec, I N);
} // namespace ubench

#endif // USPARSE_BENCHMARK_UTILS_H
