#ifndef SPGEMM_KOKKOS_SPGEMM_CORE_IMPL_HPP
#define SPGEMM_KOKKOS_SPGEMM_CORE_IMPL_HPP

#include "../common/check/hip_check.hpp"
#include "align.h"
#include "build_flag.h"
#include "spgemm_numeric.h"
#include "spgemm_symbolic.h"

#ifdef DEBUG_MODE
#include <iostream>
#endif

template <typename T, typename I>
CsrMatrix<T, I> spgemm_core(const T alpha, const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b,
                            I *row_ptr_c) {
  const double sparseness_b = static_cast<double>(matrix_b.nnz) / static_cast<double>(matrix_b.rows * matrix_b.cols);

  void *temp_buffer = nullptr;
  size_t temp_buffer_size = ALIGN_256_BYTES(sizeof(I) * matrix_b.rows * 4);
  if (sparseness_b > 0.005) {
    // create break points
    constexpr size_t CHUNK_SIZE = 2048;
    const int BPS_PER_ROW = matrix_b.cols / CHUNK_SIZE + (matrix_b.cols % CHUNK_SIZE == 0 ? 0 : 1);
    const int BP_IDX_PER_ROW = BPS_PER_ROW + 1;
    temp_buffer_size = ALIGN_256_BYTES(sizeof(I) * matrix_b.rows * BP_IDX_PER_ROW);
  }
  HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&temp_buffer), temp_buffer_size))
  // Symbolic Phase
  size_t nnz_c = 0;
  I row_nnz_max = 0;
  if (sparseness_b > 0.005) {
#ifdef DEBUG_MODE
    std::cout << "compress mode symbolic phase!" << std::endl;
#endif
    spgemm_symbolic_compress<T, I>(matrix_a, matrix_b, row_ptr_c, &nnz_c, &row_nnz_max, temp_buffer, temp_buffer_size);
  } else {
#ifdef DEBUG_MODE
    std::cout << "normal mode symbolic phase!" << std::endl;
#endif
    spgemm_symbolic<T, I>(matrix_a, matrix_b, row_ptr_c, &nnz_c, &row_nnz_max, temp_buffer, temp_buffer_size);
  }
  // Numeric Phase
  CsrMatrix<T, I> matrix_c(matrix_a.rows, matrix_b.cols, nnz_c, row_ptr_c, nullptr, nullptr);
  HIP_CHECK(hipMalloc((void **)&matrix_c.col_idx_ptr, nnz_c * sizeof(I)))
  HIP_CHECK(hipMalloc((void **)&matrix_c.values_ptr, nnz_c * sizeof(T)))
  spgemm_numeric<T, I>(alpha, matrix_a, matrix_b, row_nnz_max, matrix_c, temp_buffer, temp_buffer_size);

  HIP_CHECK(hipFree(temp_buffer));
  return matrix_c;
}

#endif // SPGEMM_KOKKOS_SPGEMM_CORE_IMPL_HPP
