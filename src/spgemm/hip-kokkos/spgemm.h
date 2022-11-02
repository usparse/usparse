#ifndef SPGEMM_HIP_KOKKOS_SPGEMM_H
#define SPGEMM_HIP_KOKKOS_SPGEMM_H

#include "../common/check/hip_check.hpp"
#include "../common/tools/set_val.hpp"
#include "api/spgemm_hip_types.h"
#include "spgemm_core_impl.hpp"

template <typename T, typename I>
void kokkos_spgemm(const T alpha, const size_t m, const size_t n, const size_t k, const size_t nnzA, const I *pOffsetA,
                   const I *pColIndexA, const T *pValueA, const size_t nnzB, const I *pOffsetB, const I *pColIndexB,
                   const T *pValueB, size_t *pNnzC, I *pOffsetC, I **ppColIndexC, T **ppValueC) {
#ifdef DEVICE_MALLOC_INITILIZE
  // HIP_CHECK(hipMemset(pOffsetC, static_cast<I>(0), (m + 1) * sizeof(I)))
  common::set_val<I, 0>(pOffsetC);
#endif
  CsrMatrix<T, I> matrix_a(m, k, nnzA, const_cast<I *>(pOffsetA), const_cast<I *>(pColIndexA),
                           const_cast<T *>(pValueA));
  CsrMatrix<T, I> matrix_b(k, n, nnzB, const_cast<I *>(pOffsetB), const_cast<I *>(pColIndexB),
                           const_cast<T *>(pValueB));
  CsrMatrix<T, I> matrix_c = spgemm_core<T, I>(alpha, matrix_a, matrix_b, pOffsetC);
  *pNnzC = matrix_c.nnz;
  *ppColIndexC = matrix_c.col_idx_ptr;
  *ppValueC = matrix_c.values_ptr;
}

#endif // SPGEMM_HIP_KOKKOS_SPGEMM_H