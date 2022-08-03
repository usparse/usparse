#ifndef SPGEMM_KOKKOS_SPGEMM_CORE_IMPL_HPP
#define SPGEMM_KOKKOS_SPGEMM_CORE_IMPL_HPP

#include "build_flag.h"
#include "csr_sort.h"
#include "error_macro.hpp"
#include "spgemm_numeric.h"
#include "spgemm_symbolic.h"

#ifdef TIME_MODE
#include "timer.h"
#include <iostream>
#endif

template <typename T, typename I>
CsrMatrix<T, I> spgemm_core(const T alpha, const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b,
                            I *row_ptr_c) {
#ifndef TIME_MODE
  // Symbolic Phase
  size_t nnz_c = 0;
  I max_row_num = 0;
  spgemm_symbolic<T, I>(matrix_a, matrix_b, row_ptr_c, &nnz_c, &max_row_num);
  // Numeric Phase
  bool is_sorted = false;
  CsrMatrix<T, I> matrix_c(matrix_a.rows, matrix_b.cols, nnz_c, row_ptr_c, nullptr, nullptr);
  KOKKOS_HIP_CHECK(hipMalloc((void **)&matrix_c.col_idx_ptr, nnz_c * sizeof(I)))
  KOKKOS_HIP_CHECK(hipMalloc((void **)&matrix_c.values_ptr, nnz_c * sizeof(T)))
  spgemm_numeric(alpha, matrix_a, matrix_b, max_row_num, matrix_c, is_sorted);
  // Sort Matrix
  if (!is_sorted) {
    csr_matrix_sort<T, I>(matrix_c);
  }
#endif

#ifdef TIME_MODE
  Timer timer;
  hipDeviceSynchronize();
  // Symbolic Phase
  timer.start();
  size_t nnz_c = 0;
  I max_row_num = 0;
  spgemm_symbolic<T, I>(matrix_a, matrix_b, row_ptr_c, &nnz_c, &max_row_num);
  hipDeviceSynchronize();
  timer.stop();
  std::cout << "Synbolic phase: " << timer.time_use.count() << "(us)" << std::endl;
  // Numeric Phase
  timer.start();
  bool is_sorted = false;
  CsrMatrix<T, I> matrix_c(matrix_a.rows, matrix_b.cols, nnz_c, row_ptr_c, nullptr, nullptr);
  KOKKOS_HIP_CHECK(hipMalloc((void **)&matrix_c.col_idx_ptr, nnz_c * sizeof(I)))
  KOKKOS_HIP_CHECK(hipMalloc((void **)&matrix_c.values_ptr, nnz_c * sizeof(T)))
  spgemm_numeric(alpha, matrix_a, matrix_b, max_row_num, matrix_c, is_sorted);
  hipDeviceSynchronize();
  timer.stop();
  std::cout << "Numeric phase: " << timer.time_use.count() << "(us)" << std::endl;
  // Sort Matrix
  timer.start();
  if (!is_sorted) {
    csr_matrix_sort<T, I>(matrix_c);
  }
  hipDeviceSynchronize();
  timer.stop();
  std::cout << "Sort phase: " << timer.time_use.count() << "(us)" << std::endl;
#endif
  return matrix_c;
}

#endif // SPGEMM_KOKKOS_SPGEMM_CORE_IMPL_HPP
