#ifndef SPGEMM_KOKKOS_SPGEMM_NUMERIC_H
#define SPGEMM_KOKKOS_SPGEMM_NUMERIC_H

#include "csr_matrix.h"
#include "spgemm_numeric_impl.hpp"

/**
 * numeric phase
 * compute C = A * B
 * alpha: input, constant coefficient
 * matrix_a: input, csr format matrix of A
 * matrix_b: input, bitmap format matrix of B
 * max_row_num: intput, max row num of csr format matrix C
 * matrix_c: output, csr format matrix of C
 * is_sorted: output, is matrix C have benn sorted
 **/
template <typename T, typename I>
void spgemm_numeric(const I alpha, const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b,
                    const I max_row_num, CsrMatrix<T, I> &matrix_c, I *bp, bool &is_sorted);

#endif // SPGEMM_KOKKOS_SPGEMM_NUMERIC_H