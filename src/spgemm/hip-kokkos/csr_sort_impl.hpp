#ifndef SPGEMM_KOKKOS_CSR_SORT_IMPL_HPP
#define SPGEMM_KOKKOS_CSR_SORT_IMPL_HPP

#include <hip/hip_runtime.h>

#include "csr_matrix.h"
#include "csr_sort_kernel.hpp"
#include "error_macro.hpp"
#include "api/sort_hip_api.h"

template <typename T, typename I> void csr_matrix_sort(CsrMatrix<T, I> &matrix) {
  csr_sort_hip<T, I>(matrix.rows, matrix.nnz, matrix.row_ptr, matrix.col_idx_ptr, matrix.values_ptr);
}

#endif // SPGEMM_KOKKOS_CSR_SORT_IMPL_HPP