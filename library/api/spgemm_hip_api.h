//
// Created by genshen on 2022/5/25.
//

#ifndef SPGEMM_HIP_API_H
#define SPGEMM_HIP_API_H

#include "spgemm_hip_types.h"

/**
 * sparse_gemm_hip calculates C <- alpha * op(A) * op(B) + beta * C, where A, B, C are sparse matrix.
 * A is a matrix of m*k, B is a matrix of k*n, and C is a matrix of m*n.
 * @tparam T  type of float point precision number.
 * @tparam I type of data index
 * @param transA: Operation op(A), can be none or transpose.
 * @param transB: Operation op(B), can be none or transpose.
 * @param alpha scalar used for multiplication.
 * @param beta scalar used for multiplication.
 * @param m number of rows of sparse matrix A and C.
 * @param n number of columns of sparse matrix B and C.
 * @param k number of columns of sparse matrix A, and number of rows of sparse matrix B.
 * @param nnzA number of non-zero elements of matrix A in CSR format
 * @param pOffsetA row offset of matrix A in CSR format.
 * @param pColIndexA column index of matrix A in CSR format.
 * @param pValueA value array of matrix A in CSR format.
 * @param nnzB number of non-zero elements of matrix B in CSR format
 * @param pOffsetB row offset of matrix B in CSR format.
 * @param pColIndexB column index of matrix B in CSR format.
 * @param pValueB value array of matrix B in CSR format.
 * @param pNnzC pointer of number of non-zero elements of matrix C in CSR format
 * @param pOffsetC row offset of matrix C in CSR format.
 * @param ppColIndexC column index of matrix C in CSR format.
 * @param ppValueC value array of matrix C in CSR format.
 */
template <typename T, typename I>
void sparse_gemm_hip(const hip_sparse_operation transA, const hip_sparse_operation transB, const T alpha, const T beta,
                     const size_t m, const size_t n, const size_t k, const size_t nnzA, const I *pOffsetA,
                     const I *pColIndexA, const T *pValueA, const size_t nnzB, const I *pOffsetB, const I *pColIndexB,
                     const T *pValueB, size_t *pNnzC, I *pOffsetC, I **ppColIndexC, T **ppValueC);

#endif // SPGEMM_HIP_API_H
