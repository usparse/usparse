#ifndef SPGEMM_KOKKOS_CSR_SORT_H
#define SPGEMM_KOKKOS_CSR_SORT_H

#include "csr_matrix.h"
#include "csr_sort_impl.hpp"
/**
 * sort each row of csr format matrix by col index
 * matrix: input/output, csr format matrix to sort
 **/
template <typename T, typename I> void csr_matrix_sort(CsrMatrix<T, I> &matrix);

#endif // SPGEMM_KOKKOS_CSR_SORT_H