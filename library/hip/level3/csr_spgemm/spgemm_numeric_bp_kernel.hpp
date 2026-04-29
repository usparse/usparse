#ifndef USTB_SPARSE_SPGEMM_SPGEMM_NUMERIC_BP_KERNEL_HPP
#define USTB_SPARSE_SPGEMM_SPGEMM_NUMERIC_BP_KERNEL_HPP

__device__ __forceinline__ int __lane_id__() {
  return threadIdx.x % __WF_SIZE__;
}

// breakpoint policy for matrixc
template <typename T, typename I, size_t CHUNK_SIZE, size_t THREADS, size_t VEC_SIZE>
__global__ void __launch_bounds__(THREADS)
    spgemm_numeric_block_per_row_sorted_bp_kernel(const T alpha, const size_t m, const size_t n, const I *row_ptr_a,
                                                  const I *col_idx_ptr_a, const T *values_ptr_a, const I *row_ptr_b,
                                                  const I *col_idx_ptr_b, const T *values_ptr_b, const I *row_ptr_c,
                                                  I *ids, T *values, const I *bp, const size_t BPS_PER_ROW,
                                                  const size_t BP_IDX_PER_ROW) {
  // block id in global
  const int global_block_id = blockIdx.x;
  // block num in global
  const int global_block_num = gridDim.x;
  // thread id in block
  const int block_thread_id = threadIdx.x;
  // thread num in block
  constexpr int block_thread_num = THREADS;
  // vec id in block
  const int block_vec_id = threadIdx.x / VEC_SIZE;
  // vec num in block
  constexpr int block_vec_num = THREADS / VEC_SIZE;
  // thread id in vec
  const int vec_thread_id = block_thread_id % VEC_SIZE;
  // thread num in vec
  constexpr int vec_thread_num = VEC_SIZE;
  // chunks
  __shared__ bool table[CHUNK_SIZE];
  __shared__ T data[CHUNK_SIZE];

  // block processes a row
  if (global_block_id >= m) {
    return;
  }
  const int row_a = global_block_id;
  const I row_idx_begin = row_ptr_a[row_a];
  const I row_idx_end = row_ptr_a[row_a + 1];
  I row_begin_c = row_ptr_c[row_a];

  for (int r = 0; r < BPS_PER_ROW; r++) {
    const int bp_idx = r;
    const int chunk_begin = bp_idx * CHUNK_SIZE;
    // init
    for (int i = block_thread_id; i < CHUNK_SIZE; i += block_thread_num) {
      table[i] = 0;
      data[i] = static_cast<T>(0);
    }
    __syncthreads();

    for (I idx_a = row_idx_begin + block_vec_id; idx_a < row_idx_end; idx_a += block_vec_num) {
      const I col_a = col_idx_ptr_a[idx_a];
      const I row_b = col_a;
      const T val_A = alpha * values_ptr_a[idx_a];

      I idx_begin_b = bp[row_b * BP_IDX_PER_ROW + bp_idx];
      I idx_end_b = bp[row_b * BP_IDX_PER_ROW + bp_idx + 1];

      for (I idx_b = idx_begin_b + vec_thread_id; idx_b < idx_end_b; idx_b += vec_thread_num) {
        const I col_b = col_idx_ptr_b[idx_b];
        table[col_b - chunk_begin] = 1;
        atomicAdd(data + col_b - chunk_begin, val_A * values_ptr_b[idx_b]);
      }
    }
    __syncthreads();
    // re-use shared memory
    int *scan_offsets = reinterpret_cast<int *>(data);
    // compress
    for (int i = block_thread_id; i < CHUNK_SIZE; i += block_thread_num) {
      bool has_nnz = table[i];
      T value = data[i];

      unsigned long long mask = __ballot(has_nnz);
      int nnz = __popcll(mask);

      unsigned long long lanemask_le = UINT64_MAX >> (sizeof(unsigned long long) * CHAR_BIT - (__lane_id__() + 1));

      int offset = __popcll(lanemask_le & mask);
      __syncthreads();

      scan_offsets[block_thread_id / __WF_SIZE__] = nnz;
      __syncthreads();

      for (unsigned int j = 1; j < block_thread_num / __WF_SIZE__; ++j) {
        if (block_thread_id >= j * __WF_SIZE__) {
          offset += scan_offsets[j - 1];
        }
      }
      I idx = row_begin_c + offset - 1;

      if (has_nnz) {
        ids[idx] = chunk_begin + i;
        values[idx] = value;
      }

      if (block_thread_id == block_thread_num - 1) {
        scan_offsets[block_thread_num / __WF_SIZE__ - 1] = offset;
      }
      __syncthreads();
      row_begin_c += scan_offsets[block_thread_num / __WF_SIZE__ - 1];
    }
    __syncthreads();
  }
}

// breakpoint policy for matrixc's group
template <typename T, typename I, size_t CHUNK_SIZE, size_t THREADS, size_t VEC_SIZE>
__global__ void __launch_bounds__(THREADS)
    spgemm_numeric_block_per_row_sorted_bp_kernel(const T alpha, const size_t m, const size_t n, const I *row_ptr_a,
                                                  const I *col_idx_ptr_a, const T *values_ptr_a, const I *row_ptr_b,
                                                  const I *col_idx_ptr_b, const T *values_ptr_b, const I *group_row_ptr,
                                                  const int group_row_num, const I *row_ptr_c, I *ids, T *values, const I *bp, 
                                                  const size_t BPS_PER_ROW, const size_t BP_IDX_PER_ROW) {
  // block id in global
  const int global_block_id = blockIdx.x;
  // block num in global
  const int global_block_num = gridDim.x;
  // thread id in block
  const int block_thread_id = threadIdx.x;
  // thread num in block
  constexpr int block_thread_num = THREADS;
  // vec id in block
  const int block_vec_id = threadIdx.x / VEC_SIZE;
  // vec num in block
  constexpr int block_vec_num = THREADS / VEC_SIZE;
  // thread id in vec
  const int vec_thread_id = block_thread_id % VEC_SIZE;
  // thread num in vec
  constexpr int vec_thread_num = VEC_SIZE;
  // chunks
  __shared__ bool table[CHUNK_SIZE];
  __shared__ T data[CHUNK_SIZE];

  // block processes a row
  if (global_block_id >= group_row_num) {
    return;
  }
  const int row_a = group_row_ptr[global_block_id];
  const I row_idx_begin = row_ptr_a[row_a];
  const I row_idx_end = row_ptr_a[row_a + 1];
  I row_begin_c = row_ptr_c[row_a];

  for (int r = 0; r < BPS_PER_ROW; r++) {
    const int bp_idx = r;
    const int chunk_begin = bp_idx * CHUNK_SIZE;
    // init
    for (int i = block_thread_id; i < CHUNK_SIZE; i += block_thread_num) {
      table[i] = 0;
      data[i] = static_cast<T>(0);
    }
    __syncthreads();

    for (I idx_a = row_idx_begin + block_vec_id; idx_a < row_idx_end; idx_a += block_vec_num) {
      const I col_a = col_idx_ptr_a[idx_a];
      const I row_b = col_a;
      const T val_A = alpha * values_ptr_a[idx_a];

      I idx_begin_b = bp[row_b * BP_IDX_PER_ROW + bp_idx];
      I idx_end_b = bp[row_b * BP_IDX_PER_ROW + bp_idx + 1];

      for (I idx_b = idx_begin_b + vec_thread_id; idx_b < idx_end_b; idx_b += vec_thread_num) {
        const I col_b = col_idx_ptr_b[idx_b];
        table[col_b - chunk_begin] = 1;
        atomicAdd(data + col_b - chunk_begin, val_A * values_ptr_b[idx_b]);
      }
    }
    __syncthreads();
    // re-use shared memory
    int *scan_offsets = reinterpret_cast<int *>(data);
    // compress
    for (int i = block_thread_id; i < CHUNK_SIZE; i += block_thread_num) {
      bool has_nnz = table[i];
      T value = data[i];

      unsigned long long mask = __ballot(has_nnz);
      int nnz = __popcll(mask);

      unsigned long long lanemask_le = UINT64_MAX >> (sizeof(unsigned long long) * CHAR_BIT - (__lane_id() + 1));

      int offset = __popcll(lanemask_le & mask);
      __syncthreads();

      scan_offsets[block_thread_id / __WF_SIZE__] = nnz;
      __syncthreads();

      for (unsigned int j = 1; j < block_thread_num / __WF_SIZE__; ++j) {
        if (block_thread_id >= j * __WF_SIZE__) {
          offset += scan_offsets[j - 1];
        }
      }
      I idx = row_begin_c + offset - 1;

      if (has_nnz) {
        ids[idx] = chunk_begin + i;
        values[idx] = value;
      }

      if (block_thread_id == block_thread_num - 1) {
        scan_offsets[block_thread_num / __WF_SIZE__ - 1] = offset;
      }
      __syncthreads();
      row_begin_c += scan_offsets[block_thread_num / __WF_SIZE__ - 1];
    }
    __syncthreads();
  }
}

#endif // USTB_SPARSE_SPGEMM_SPGEMM_NUMERIC_BP_KERNEL_HPP
