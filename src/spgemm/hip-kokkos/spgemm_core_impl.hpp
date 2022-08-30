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
  constexpr size_t CHUNK_SIZE = 2048;
  const int BPS_PER_ROW = matrix_b.cols / CHUNK_SIZE + (matrix_b.cols % CHUNK_SIZE == 0 ? 0 : 1);
  const int BP_IDX_PER_ROW = BPS_PER_ROW + 1;

  I *temp_buffer = nullptr;
  const size_t temp_buffer_size = sizeof(I) * matrix_b.rows * BP_IDX_PER_ROW;
  KOKKOS_HIP_CHECK(hipMalloc((void **)&temp_buffer, temp_buffer_size))
  // Symbolic Phase
  size_t nnz_c = 0;
  I max_row_num = 0;
  spgemm_symbolic<T, I>(matrix_a, matrix_b, row_ptr_c, &nnz_c, &max_row_num, temp_buffer, temp_buffer_size);
  // Numeric Phase
  CsrMatrix<T, I> matrix_c(matrix_a.rows, matrix_b.cols, nnz_c, row_ptr_c, nullptr, nullptr);
  KOKKOS_HIP_CHECK(hipMalloc((void **)&matrix_c.col_idx_ptr, nnz_c * sizeof(I)))
  KOKKOS_HIP_CHECK(hipMalloc((void **)&matrix_c.values_ptr, nnz_c * sizeof(T)))
  spgemm_numeric(alpha, matrix_a, matrix_b, max_row_num, matrix_c, temp_buffer, temp_buffer_size);

  KOKKOS_HIP_CHECK(hipFree(temp_buffer));
  return matrix_c;
}

#endif // SPGEMM_KOKKOS_SPGEMM_CORE_IMPL_HPP
