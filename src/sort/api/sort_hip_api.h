//
// Created by yuanjie on 2022/6/23.
//

#ifndef SORT_HIP_API_H
#define SORT_HIP_API_H

#include "sort_hip_types.h"

/**
 * csr_sort_hip sort a sparse matrix in CSR format.
 * @tparam T type of float point precision number.
 * @tparam I type of data index.
 * @param m number of rows of sparse matrix.
 * @param nnz number of non-zero elements of matrix in CSr format.
 * @param pOffset row offset of matrix in CSR format.
 * @param pColIndex column index of matrix in CSR format.
 * @param pValue value array of matrix in CSR format.
 */
template <typename T, typename I>
void csr_sort_hip(const size_t m, const size_t nnz, const I *pOffset, I *pColIndex, T *pValue);

#endif // SORT_HIP_API_H
