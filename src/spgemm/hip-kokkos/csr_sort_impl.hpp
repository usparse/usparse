#ifndef SPGEMM_KOKKOS_CSR_SORT_IMPL_HPP
#define SPGEMM_KOKKOS_CSR_SORT_IMPL_HPP

#include "csr_matrix.h"
#include "csr_sort_kernel.hpp"
#include "error_macro.hpp"
#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>
#include <rocsparse.h>

template <typename T, typename I> void rocsparse_sort_wrapper(CsrMatrix<T, I> &matrix) {
  using dtype = T;
  using csrIdxType = I;
  // Create rocsparse handle
  rocsparse_handle handle;
  ROCSPARSE_CHECK(rocsparse_create_handle(&handle));
  // Create rocsparse matrix descr
  rocsparse_mat_descr descr;
  ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr))

  // Create permutation vector perm as the identity map
  rocsparse_int *perm;
  KOKKOS_HIP_CHECK(hipMalloc((void **)&perm, sizeof(rocsparse_int) * matrix.nnz))
  ROCSPARSE_CHECK(rocsparse_create_identity_permutation(handle, matrix.nnz, perm))

  // Allocate temporary buffer
  size_t buffer_size;
  void *temp_buffer;
  ROCSPARSE_CHECK(rocsparse_csrsort_buffer_size(handle, matrix.rows, matrix.cols, matrix.nnz, matrix.row_ptr,
                                                matrix.col_idx_ptr, &buffer_size))
  KOKKOS_HIP_CHECK(hipMalloc(&temp_buffer, buffer_size))

  // Sort the CSR matrix
  ROCSPARSE_CHECK(rocsparse_csrsort(handle, matrix.rows, matrix.cols, matrix.nnz, descr, matrix.row_ptr,
                                    matrix.col_idx_ptr, perm, temp_buffer))

  // Free buffer
  KOKKOS_HIP_CHECK(hipFree(temp_buffer))
  ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr))
  ROCSPARSE_CHECK(rocsparse_destroy_handle(handle))
  // Gather sorted csr_val array
  dtype *csr_val_sorted;
  KOKKOS_HIP_CHECK(hipMalloc((void **)&csr_val_sorted, sizeof(dtype) * matrix.nnz));
  densetosparse_gthr_wrapper<dtype>(matrix.nnz, matrix.values_ptr, csr_val_sorted, perm, rocsparse_index_base_zero);

  // Clean up
  KOKKOS_HIP_CHECK(hipFree(perm))
  KOKKOS_HIP_CHECK(hipFree(matrix.values_ptr))

  // Reset values ptr
  matrix.values_ptr = csr_val_sorted;
}

template <typename T, typename I> void rocprim_sort_wrapper(CsrMatrix<T, I> &matrix) {
  size_t buffer_size;
  void *buffer = nullptr;
  unsigned int startbit = 0;
  unsigned int endbit = 64 - __builtin_clzll(matrix.cols);

  rocprim::segmented_radix_sort_pairs(buffer, buffer_size, matrix.col_idx_ptr, matrix.col_idx_ptr, matrix.values_ptr,
                                      matrix.values_ptr, matrix.nnz, matrix.rows, matrix.row_ptr, matrix.row_ptr + 1,
                                      startbit, endbit);

  KOKKOS_HIP_CHECK(hipMalloc((void **)&buffer, buffer_size))

  rocprim::segmented_radix_sort_pairs(buffer, buffer_size, matrix.col_idx_ptr, matrix.col_idx_ptr, matrix.values_ptr,
                                      matrix.values_ptr, matrix.nnz, matrix.rows, matrix.row_ptr, matrix.row_ptr + 1,
                                      startbit, endbit);
  KOKKOS_HIP_CHECK(hipFree(buffer))
}

template <typename T, typename I> void csr_matrix_sort(CsrMatrix<T, I> &matrix) {
  // rocprim_sort_wrapper<T, I>(matrix);
  rocsparse_sort_wrapper<T, I>(matrix);
}

#endif // SPGEMM_KOKKOS_CSR_SORT_IMPL_HPP