//
// Created by genshen on 2022/5/25.
//

#ifndef SPGEMM_HIP_API_H
#define SPGEMM_HIP_API_H

enum sparse_operation { operation_none = 0, operation_transpose = 1 };

typedef float sp_data_type;     // type of single precision float point number.
typedef double dp_data_type;    // type of double precision float point number.
typedef dp_data_type data_type; // default data type for float point number.
typedef int idx_type;           // type of index.

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
 * @param pOffsetA row offset of matrix A in CSR format.
 * @param pColIndexA column index of matrix A in CSR format.
 * @param pValueA value array of matrix A in CSR format.
 * @param pOffsetB row offset of matrix B in CSR format.
 * @param pColIndexB column index of matrix B in CSR format.
 * @param pValueB value array of matrix B in CSR format.
 * @param pValueC value array of matrix C in CSR format.
 */
template <typename T, typename I>
void sparse_gemm_hip(const sparse_operation transA, const sparse_operation transB, const T alpha, const T beta,
                     const I m, const I n, const I k, const I *pOffsetA, const I *pColIndexA, const T *pValueA,
                     const I *pOffsetB, const I *pColIndexB, const T *pValueB, T *pValueC);

#endif // SPGEMM_HIP_API_H
