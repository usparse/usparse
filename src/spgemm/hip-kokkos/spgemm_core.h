#ifndef SPGEMM_KOKKOS_SPGEMM_CORE_H
#define SPGEMM_KOKKOS_SPGEMM_CORE_H

#include "csr_matrix.h"
#include "spgemm_core_impl.hpp"

/**
 * calculate C = alpha * A * B
 * alpha: input, constant coefficient
 * matrix_a: input, csr format matrix of A
 * matrix_b: input, csr format matrix of B
 * row_ptr_c: input/output, row ptr of matrix C
 * return: csr format of matrix C
 * tips: memory of matrix(A|B|C) must in device
 **/
template <typename T, typename I>
CsrMatrix<T, I> spgemm_core(const T alpha, const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b,
                            I *row_ptr_c);

#endif // SPGEMM_KOKKOS_SPGEMM_CORE_H
