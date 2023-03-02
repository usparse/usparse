#ifndef USTB_SPARSE_SPGEMM_SPGEMM_SYMBOLIC_SINGLE_ELEMENT_KERNEL_HPP
#define USTB_SPARSE_SPGEMM_SPGEMM_SYMBOLIC_SINGLE_ELEMENT_KERNEL_HPP

template <typename T, typename I>
__global__ void spgemm_symbolic_thread_per_row_single_element_kernel(const size_t m, const I *row_ptr_a,
                                                                        const I *col_idx_ptr_a, const I *row_ptr_b,
                                                                        const I *col_idx_ptr_b, const I *group_row_ptr, 
                                                                        const int group_row_num, I *row_ptr_c) {
  // thread id in global
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  // thread num in global
  const int global_thread_num = gridDim.x * blockDim.x;

  if (global_thread_id >= group_row_num) {
    return;
  }
  
  const int row_a = group_row_ptr[global_thread_id];
  const int row_index_begin = row_ptr_a[row_a];
  const I col_a = col_idx_ptr_a[row_index_begin];
  const I row_b = col_a;
  row_ptr_c[row_a + 1] = row_ptr_b[row_b + 1] - row_ptr_b[row_b];
}

#endif // USTB_SPARSE_SPGEMM_SPGEMM_SYMBOLIC_SINGLE_ELEMENT_KERNEL_HPP