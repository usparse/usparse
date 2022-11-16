//
// Created by genshen on 2022/5/25.
//

#include "spgemm_hip_api.h"
#include "building_config.h"

#include "src/spgemm.h"

template <typename T, typename I>
void sparse_gemm_hip(const hip_sparse_operation transA, const hip_sparse_operation transB, 
                     const T alpha, const T beta, const size_t m, const size_t n, const size_t k, 
                     const size_t nnzA, const I *pOffsetA, const I *pColIndexA, const T *pValueA,
                     const size_t nnzB, const I *pOffsetB, const I *pColIndexB, const T *pValueB, 
                     size_t *pNnzC, I *pOffsetC, I **ppColIndexC, T **ppValueC) {

    kokkos_spgemm<T, I>(alpha, m, n, k, nnzA, pOffsetA, pColIndexA, pValueA, 
                      nnzB, pOffsetB, pColIndexB, pValueB, 
                      pNnzC, pOffsetC, ppColIndexC, ppValueC);
}

// template instantiation
template void sparse_gemm_hip<dp_data_type, idx_type>(const hip_sparse_operation transA, const hip_sparse_operation transB, 
                                                      const dp_data_type alpha, const dp_data_type beta, const size_t m, const size_t n, const size_t k, 
                                                      const size_t nnzA, const idx_type *pOffsetA, const idx_type *pColIndexA, const dp_data_type *pValueA,
                                                      const size_t nnzB, const idx_type *pOffsetB, const idx_type *pColIndexB, const dp_data_type *pValueB, 
                                                      size_t *pNnzC, idx_type *pOffsetC, idx_type **ppColIndexC, dp_data_type **ppValueC);

template void sparse_gemm_hip<sp_data_type, idx_type>(const hip_sparse_operation transA, const hip_sparse_operation transB, 
                                                      const sp_data_type alpha, const sp_data_type beta, const size_t m, const size_t n, const size_t k, 
                                                      const size_t nnzA, const idx_type *pOffsetA, const idx_type *pColIndexA, const sp_data_type *pValueA,
                                                      const size_t nnzB, const idx_type *pOffsetB, const idx_type *pColIndexB, const sp_data_type *pValueB, 
                                                      size_t *pNnzC, idx_type *pOffsetC, idx_type **ppColIndexC, sp_data_type **ppValueC);
