#ifndef SPGEMM_KOKKOS_SPGEMM_SYMBOLIC_H
#define SPGEMM_KOKKOS_SPGEMM_SYMBOLIC_H

#include "csr_matrix.h"
#include "spgemm_symbolic_impl.hpp"

/**
 * symbolic phase
 * compute row ptr & nnz of matrix C = A * B
 * matrix_a: input, csr format matrix of A
 * matrix_b: input, csr format matrix of B
 * row_ptr_c: output, row ptr of csr format matrix C
 * nnz_c: output, nonzero number count of csr format matrix C
 * max_row_num: output, max row num of csr format matrix C
 **/
template <typename T, typename I>
void spgemm_symbolic(const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b, I *row_ptr_c, size_t *nnz_c,
                     I *max_row_num, I *bp);

#endif // SPGEMM_KOKKOS_SPGEMM_SYMBOLIC_H