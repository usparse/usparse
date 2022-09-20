#ifndef SPGEMM_KOKKOS_SPGEMM_CORE_IMPL_HPP
#define SPGEMM_KOKKOS_SPGEMM_CORE_IMPL_HPP

#include "../common/check/hip_check.hpp"
#include "align.h"
#include "spgemm_numeric_impl.hpp"
#include "spgemm_symbolic_impl.hpp"

template <typename T, typename I>
CsrMatrix<T, I> spgemm_core(const T alpha, const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b,
                            I *row_ptr_c) {
  const double sparseness_b = static_cast<double>(matrix_b.nnz) / static_cast<double>(matrix_b.rows * matrix_b.cols);

  // alloc temp buffer
  void *temp_buffer = nullptr;
  size_t temp_buffer_size = 0;
  size_t symbolic_buffer_size = 0;
  size_t numeric_buffer_size = 0;

  bool is_compress = (sparseness_b < 0.005 && matrix_b.cols > 30000) ? false : true;
  bool is_use_bp = sparseness_b > 0.005 ? true : false;

  if (is_compress) {
    symbolic_buffer_size = spgemm_symbolic_compress_buffer_size<T, I>(matrix_a, matrix_b);
  } else {
    symbolic_buffer_size = spgemm_symbolic_buffer_size<T, I>(matrix_a, matrix_b);
  }

  if (is_use_bp) {
    numeric_buffer_size = spgemm_numeric_buffer_size<T, I, true>(matrix_a, matrix_b);
  } else {
    numeric_buffer_size = spgemm_numeric_buffer_size<T, I, false>(matrix_a, matrix_b);
  }
  temp_buffer_size = std::max<size_t>(symbolic_buffer_size, numeric_buffer_size);
  HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&temp_buffer), temp_buffer_size))

  // symbolic Phase
  size_t nnz_c = 0;
  I row_nnz_max = 0;
  if (is_compress) {
    spgemm_symbolic_compress<T, I>(matrix_a, matrix_b, row_ptr_c, &nnz_c, &row_nnz_max, temp_buffer);
  } else {
    spgemm_symbolic<T, I>(matrix_a, matrix_b, row_ptr_c, &nnz_c, &row_nnz_max, temp_buffer);
  }
  CsrMatrix<T, I> matrix_c(matrix_a.rows, matrix_b.cols, nnz_c, row_ptr_c, nullptr, nullptr);
  HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&matrix_c.col_idx_ptr), nnz_c * sizeof(I)))
  HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&matrix_c.values_ptr), nnz_c * sizeof(T)))
  // numeric Phase
  if (is_use_bp) {
    spgemm_numeric<T, I, true>(alpha, matrix_a, matrix_b, row_nnz_max, matrix_c, temp_buffer);
  } else {
    spgemm_numeric<T, I, false>(alpha, matrix_a, matrix_b, row_nnz_max, matrix_c, temp_buffer);
  }
  // free temp memory
  HIP_CHECK(hipFree(temp_buffer));
  return matrix_c;
}

#endif // SPGEMM_KOKKOS_SPGEMM_CORE_IMPL_HPP
