//
// Created by genshen on 2022/5/25.
//

#include "spgemm_hip_api.h"
#include "building_config.h"

template <typename T, typename I>
void sparse_gemm_hip(const sparse_operation transA, const sparse_operation transB, const T alpha, const T beta,
                     const I m, const I n, const I k, const I *pOffsetA, const I *pColIndexA, const T *pValueA,
                     const I *pOffsetB, const I *pColIndexB, const T *pValueB, T *pValueC) {
  // todo:
}

// template instantiation
template void sparse_gemm_hip<dp_data_type, idx_type>(const sparse_operation transA, const sparse_operation transB,
                                                      const dp_data_type alpha, const dp_data_type beta,
                                                      const idx_type m, const idx_type n, const idx_type k,
                                                      const idx_type *pOffsetA, const idx_type *pColIndexA,
                                                      const dp_data_type *pValueA, const idx_type *pOffsetB,
                                                      const idx_type *pColIndexB, const dp_data_type *pValueB,
                                                      dp_data_type *pValueC);

template void sparse_gemm_hip<sp_data_type, idx_type>(const sparse_operation transA, const sparse_operation transB,
                                                      const sp_data_type alpha, const sp_data_type beta,
                                                      const idx_type m, const idx_type n, const idx_type k,
                                                      const idx_type *pOffsetA, const idx_type *pColIndexA,
                                                      const sp_data_type *pValueA, const idx_type *pOffsetB,
                                                      const idx_type *pColIndexB, const sp_data_type *pValueB,
                                                      sp_data_type *pValueC);
