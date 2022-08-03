#ifndef SPGEMM_KOKKOS_SPGEMM_CORE_H
#define SPGEMM_KOKKOS_SPGEMM_CORE_H

#include "compress_matrix.h"
#include "csr_matrix.h"
#include "spgemm_core.hpp"

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

/**
 * compress csr format matrix to bitmap format
 * input: csr format matrix
 * return: compress format matrix
 **/
template <typename T, typename I> CompressMatrix<I> compress_matrix(const CsrMatrix<T, I> &csr_matrix);

/**
 * symbolic phase
 * compute row ptr & nnz of matrix C = A * B
 * matrix_a: input, csr format matrix of A
 * bitset: input, compress format colindex of B
 * row_ptr_c: output, row ptr of csr format matrix C
 * nnz_c: output, nonzero number count of csr format matrix C
 **/
template <typename T, typename I>
void spgemm_symbolic(const CsrMatrix<T, I> &matrix_a, const CompressMatrix<I> &bitset_b, I *row_ptr_c, size_t *nnz_c);

/**
 * numeric phase
 * compute C = A * B
 * alpha: input, constant coefficient
 * matrix_a: input, csr format matrix of A
 * matrix_b: input, bitmap format matrix of B
 * matrix_c: output, csr format matrix of C
 **/
template <typename T, typename I>
void spgemm_numeric(const I alpha, const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b,
                    CsrMatrix<T, I> &matrix_c);

/**
 * sort each row of csr format matrix by col index
 * matrix: input/output, csr format matrix to sort
 **/
template <typename T, typename I> void csr_matrix_sort(CsrMatrix<T, I> &matrix);

#endif // SPGEMM_KOKKOS_SPGEMM_CORE_H
