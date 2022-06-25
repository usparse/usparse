#ifndef HIP_BLOCK_RADIX_CSR_SORT_H
#define HIP_BLOCK_RADIX_CSR_SORT_H

#include <hip/hip_runtime.h>
#include "block_radix_sort.h"

template <typename T, typename I>
void csr_sort(size_t m, size_t nnz, const I* row_ptr, I* col_idx_ptr, T* values_ptr) {
  I *col_src_data, *col_temp_data, *row_src_data;
  T *value_src_data, *value_temp_data;
  const int thread_num = m;
  const int block_dim = 256;
  const int block_num = thread_num / block_dim + (thread_num % block_dim == 0 ? 0 : 1);
  hipMalloc((void **)&col_temp_data, sizeof(I) * nnz);
  hipMalloc((void **)&value_temp_data, sizeof(T) * nnz);
  (block_radix_sort_kernel<T, I, sizeof(I) * 8>)<<<block_num, block_dim>>>(thread_num, row_ptr, col_idx_ptr, col_temp_data, values_ptr, value_temp_data);
  hipFree(col_temp_data);
  hipFree(value_temp_data);
}

#endif //HIP_BLOCK_RADIX_CSR_SORT_H
