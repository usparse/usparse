//
// Created by yuanjie on 2022/6/23.
//

#include "sort_hip_api.h"
#include "building_config.h"

#include "hip-block-radix/csr_sort.h"

template <typename T, typename I>
void csr_sort_hip(const size_t m, const size_t nnz, const I *pOffset, I *pColIndex, T *pValue) {
#ifdef SORT_KERNEL_STRATEGY_BLOCK_RADIX
    csr_sort<T, I>(m, nnz, pOffset, pColIndex, pValue);
#endif
}

// template instantiation
template void csr_sort_hip<dp_data_type, idx_type>(const size_t m, const size_t nnz, const idx_type *pOffset, idx_type *pColIndex, dp_data_type *pValue);

template void csr_sort_hip<sp_data_type, idx_type>(const size_t m, const size_t nnz, const idx_type *pOffset, idx_type *pColIndex, sp_data_type *pValue);
