#ifndef USTB_SPARSE_SPGEMM_SPGEMM_CORE_IMPL_HPP
#define USTB_SPARSE_SPGEMM_SPGEMM_CORE_IMPL_HPP

#include "../common/check/hip_check.hpp"
#include "../common/timer/timer.hpp"
#include "align.h"
#include "spgemm_adapt_policy.hpp"
#include "spgemm_buffer_impl.hpp"

template <typename T, typename I>
CsrMatrix<T, I> spgemm_core(const T alpha, const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b,
                            I *row_ptr_c) {
  // Start symbolic timer
  common::timer[ustb_spgemm_symbolic_timer].start();
  PolicyFlag<T, I> policyflag(matrix_a, matrix_b);
  size_t nnz_c = 0;
  I row_nnz_max = 0;

  // alloc temp buffer
  void *temp_buffer = nullptr;
  size_t temp_buffer_size = spgemm_buffer_size<T, I>(policyflag, matrix_a, matrix_b);
  HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&temp_buffer), temp_buffer_size))

  // symbolic phase
  AdaptPolicyChoice::template symbolic_stage<T, I>(policyflag, matrix_a, matrix_b, row_ptr_c, &nnz_c, &row_nnz_max,
                                                   temp_buffer);
  // Stop symbolic timer
  common::timer[ustb_spgemm_symbolic_timer].stop();

  // alloc matrix_c mem
  CsrMatrix<T, I> matrix_c(matrix_a.rows, matrix_b.cols, nnz_c, row_ptr_c, nullptr, nullptr);
  char *matrix_c_buffer_ptr = nullptr;
  HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&matrix_c_buffer_ptr),
                      ALIGN_256_BYTES(nnz_c * sizeof(I)) + ALIGN_256_BYTES(nnz_c * sizeof(T))))
  matrix_c.col_idx_ptr = reinterpret_cast<I *>(matrix_c_buffer_ptr);
  matrix_c.values_ptr = reinterpret_cast<T *>(matrix_c_buffer_ptr + ALIGN_256_BYTES(nnz_c * sizeof(I)));
  // #ifdef DEVICE_MALLOC_INITILIZE
  //   // HIP_CHECK(hipMemset(matrix_c.col_idx_ptr, static_cast<I>(0), nnz_c * sizeof(I)))
  //   // HIP_CHECK(hipMemset(matrix_c.values_ptr, static_cast<T>(0), nnz_c * sizeof(T)))
  // #endif

  // Start numeric timer
  common::timer[ustb_spgemm_numeric_timer].start();
  // numeric phase
  AdaptPolicyChoice::template numeric_stage<T, I>(policyflag, alpha, matrix_a, matrix_b, row_nnz_max, matrix_c,
                                                  temp_buffer, temp_buffer_size);

  // free temp memory
  HIP_CHECK(hipFree(temp_buffer));
  // Stop numeric timer
  common::timer[ustb_spgemm_numeric_timer].stop();
  return matrix_c;
}

#endif // USTB_SPARSE_SPGEMM_SPGEMM_CORE_IMPL_HPP
