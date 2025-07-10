#ifndef USTB_SPARSE_SPGEMM_SPGEMM_NUMERIC_SINGLE_ELEMENT_KERNEL_HPP
#define USTB_SPARSE_SPGEMM_SPGEMM_NUMERIC_SINGLE_ELEMENT_KERNEL_HPP

template <typename T, typename I, size_t THREADS, size_t VEC_SIZE>
__global__ void __launch_bounds__(THREADS) spgemm_numeric_vector_per_row_single_element_kernel(
    const T alpha, const size_t m, const size_t n, const I *row_ptr_a, const I *col_idx_ptr_a, const T *values_ptr_a,
    const I *row_ptr_b, const I *col_idx_ptr_b, const T *values_ptr_b, const I *group_row_ptr, const int group_row_num,
    const I *row_ptr_c, I *ids, T *values) {

  // thread id in global
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  // thread num in global
  const int global_thread_num = gridDim.x * blockDim.x;
  // vec id in global
  const int global_vec_id = global_thread_id / VEC_SIZE;
  // vec num in global
  const int global_vec_num = global_thread_num / VEC_SIZE;
  // thread num in block
  constexpr int block_thread_num = THREADS;
  // vec id in block
  const int block_vec_id = threadIdx.x / VEC_SIZE;
  // vec num in block
  constexpr int block_vec_num = THREADS / VEC_SIZE;
  // thread id in vec
  const int vec_thread_id = global_thread_id % VEC_SIZE;
  // thread num in vec
  constexpr int vec_thread_num = VEC_SIZE;

  if (global_vec_id >= group_row_num) {
    return;
  }
  const int row_a = group_row_ptr[global_vec_id];

  const I row_idx_begin = row_ptr_a[row_a];
  const I row_idx_end = row_ptr_a[row_a + 1];
  I *vec_mem_ids = ids + row_ptr_c[row_a];
  T *vec_mem_values = values + row_ptr_c[row_a];
  if(vec_thread_id == 0) {
      vec_mem_values[0] = 0;
  }

  for(int idx_a = row_idx_begin; idx_a < row_idx_end; ++idx_a) {
    const I col_a = col_idx_ptr_a[idx_a];
    const T val_A = values_ptr_a[idx_a];

    const I row_b = col_a;
    const I idx_begin_b = row_ptr_b[row_b];
    const I idx_end_b = row_ptr_b[row_b + 1];

    for(int idx_b = idx_begin_b + vec_thread_id; idx_b < idx_end_b; idx_b += vec_thread_num) {
        const T tmp_val = alpha * val_A * values_ptr_b[idx_b];
        const I col_b = col_idx_ptr_b[idx_b];

        vec_mem_ids[0] = col_b;
        atomicAdd(&vec_mem_values[0], tmp_val); 
    }
    // vec_mem_values[0] *= alpha;
  }
}

#endif // USTB_SPARSE_SPGEMM_SPGEMM_NUMERIC_SINGLE_ELEMENT_KERNEL_HPP