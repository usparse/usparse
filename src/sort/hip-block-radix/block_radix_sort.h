#ifndef SORT_BLOCK_RADIX_SORT_H
#define SORT_BLOCK_RADIX_SORT_H

#include <hip/hip_runtime.h>

template<typename T, typename I, size_t BitsLen = sizeof(I) * 8>
__global__ void block_radix_sort_kernel(const int rows, const I *row_ptr, I *col_idx_ptr, I *col_temp_arr, T *value_ptr, T *value_temp_arr) {
  // thread id in global
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  if (global_thread_id < rows) { 
    int begin_idx = row_ptr[global_thread_id];
    int end_idx = row_ptr[global_thread_id + 1];
    for (int bit = 0; bit < BitsLen; ++bit) {
      int bit_mask = (1 << bit);
      int count_0 = 0;
      int count_1 = 0;
      for (int i = begin_idx; i < end_idx; ++i) {
        I temp = col_idx_ptr[i];
        if (temp & bit_mask) {
          col_temp_arr[begin_idx + count_1] = col_idx_ptr[i];
          value_temp_arr[begin_idx + count_1] = value_ptr[i];
          count_1 += 1;
        } else {
          col_idx_ptr[begin_idx + count_0] = col_idx_ptr[i];
          value_ptr[begin_idx + count_0] = value_ptr[i];
          count_0 += 1;
        }
      }
      for (int j = 0; j < count_1; j++) {
        col_idx_ptr[begin_idx + count_0 + j] = col_temp_arr[begin_idx + j];
        value_ptr[begin_idx + count_0 + j] = value_temp_arr[begin_idx + j];
      }
    }
  }
}

#endif //SORT_BLOCK_RADIX_SORT_H
