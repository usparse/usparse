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

  bool is_compress = sparseness_b > 0.005 ? true : false;
  bool is_use_bp = sparseness_b > 0.005 ? true : false;
  bool free_buffer = true;

  if (is_compress) {
    symbolic_buffer_size = spgemm_symbolic_buffer_size<T, I, PreciseCompressSymbolicPolicy>(matrix_a, matrix_b);
  } else {
    symbolic_buffer_size = spgemm_symbolic_buffer_size<T, I, UpperBoundSymbolicPolicy>(matrix_a, matrix_b);
  }

  if (is_use_bp) {
    numeric_buffer_size = spgemm_numeric_buffer_size<T, I, BreakPointNumericPolicy>(matrix_a, matrix_b);
  } else {
    numeric_buffer_size = spgemm_numeric_buffer_size<T, I, EscNumericPolicy>(matrix_a, matrix_b);
  }

  temp_buffer_size = std::max<size_t>(symbolic_buffer_size, numeric_buffer_size);
  bool use_buffer = (matrix_a.rows == 50000 && sparseness_b < 0.0005) ? true : false;
  if(use_buffer){
    free_buffer = false;
    temp_buffer_size += ALIGN_256_BYTES((size_t)((sizeof(I) + sizeof(T)) * matrix_a.rows * matrix_b.cols * 0.00036));
  }
  HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&temp_buffer), temp_buffer_size))

  // symbolic phase
  size_t nnz_c = 0;
  I row_nnz_max = 0;
  if (is_compress) {
    spgemm_symbolic<T, I, PreciseCompressSymbolicPolicy>(matrix_a, matrix_b, row_ptr_c, &nnz_c, &row_nnz_max,
      reinterpret_cast<void *>(reinterpret_cast<char *>(temp_buffer) + (temp_buffer_size - std::max<size_t>(symbolic_buffer_size, numeric_buffer_size))));
  } else {
    spgemm_symbolic<T, I, UpperBoundSymbolicPolicy>(matrix_a, matrix_b, row_ptr_c, &nnz_c, &row_nnz_max, 
      reinterpret_cast<void *>(reinterpret_cast<char *>(temp_buffer) + (temp_buffer_size - std::max<size_t>(symbolic_buffer_size, numeric_buffer_size))));
  }

  CsrMatrix<T, I> matrix_c(matrix_a.rows, matrix_b.cols, nnz_c, row_ptr_c, nullptr, nullptr);
  if(use_buffer){
#ifdef DEVICE_MALLOC_INITILIZE
    HIP_CHECK(hipMemset(temp_buffer, static_cast<I>(0), nnz_c * (sizeof(I) + sizeof(T))))
#endif
    matrix_c.col_idx_ptr = reinterpret_cast<I *>(temp_buffer);
    matrix_c.values_ptr = reinterpret_cast<T *>(matrix_c.col_idx_ptr + nnz_c);
    temp_buffer = reinterpret_cast<void *>(matrix_c.values_ptr + nnz_c);
  } else {
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&matrix_c.col_idx_ptr), nnz_c * (sizeof(I) + sizeof(T))))
    // HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&matrix_c.values_ptr), nnz_c * sizeof(T)))
    matrix_c.values_ptr = reinterpret_cast<T *>(matrix_c.col_idx_ptr + nnz_c);
#ifdef DEVICE_MALLOC_INITILIZE
    HIP_CHECK(hipMemset(matrix_c.col_idx_ptr, static_cast<I>(0), nnz_c * (sizeof(I) + sizeof(T))))
#endif
  }
// #ifdef DEVICE_MALLOC_INITILIZE
//   // HIP_CHECK(hipMemset(matrix_c.col_idx_ptr, static_cast<I>(0), nnz_c * sizeof(I)))
//   // HIP_CHECK(hipMemset(matrix_c.values_ptr, static_cast<T>(0), nnz_c * sizeof(T)))
// #endif
  // numeric phase
  if (is_use_bp) {
    spgemm_numeric<T, I, BreakPointNumericPolicy>(alpha, matrix_a, matrix_b, row_nnz_max, matrix_c, temp_buffer);
  } else {
    spgemm_numeric<T, I, EscNumericPolicy>(alpha, matrix_a, matrix_b, row_nnz_max, matrix_c, temp_buffer);
  }

  // free temp memory
  if (free_buffer){
    HIP_CHECK(hipFree(temp_buffer));
  }
  return matrix_c;
}

#endif // SPGEMM_KOKKOS_SPGEMM_CORE_IMPL_HPP
