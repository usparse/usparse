#ifndef SPGEMM_KOKKOS_SPGEMM_COMPRESS_KERNEL_HPP
#define SPGEMM_KOKKOS_SPGEMM_COMPRESS_KERNEL_HPP

template <typename I, size_t BitsLen>
__global__ void compress_matrix_thread_per_row_kernel(const size_t m, const I *row_ptr, const I *col_idx_ptr,
                                                      typename CompressMatrix<I>::IdType *ids,
                                                      typename CompressMatrix<I>::BitSetEleType *bits_cols,
                                                      int *use_count) {
  using idtype_t = typename CompressMatrix<I>::IdType;
  using bittype_t = typename CompressMatrix<I>::BitSetEleType;

  // thread id in global
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  // thread num in global
  const int global_thread_num = gridDim.x * blockDim.x;
  // each thread process a row
  for (int row = global_thread_id; row < m; row += global_thread_num) {
    const I row_idx_begin = row_ptr[row];
    const I row_idx_end = row_ptr[row + 1];

    idtype_t *thread_mem_ids = ids + row_idx_begin;
    bittype_t *thread_mem_bits = bits_cols + row_idx_begin;
    idtype_t count = -1;
    idtype_t id = 0xffffffff;
    for (int idx = row_idx_begin; idx < row_idx_end; idx++) {
      const I col = col_idx_ptr[idx];
      const idtype_t my_id = col / BitsLen;
      const bittype_t my_bit = 0x1 << (col % BitsLen);
      if (my_id == id) {
        thread_mem_bits[count] |= my_bit;
      } else {
        count++;
        thread_mem_ids[count] = my_id;
        thread_mem_bits[count] = my_bit;
        id = my_id;
      }
    }
    use_count[row] = count + 1;
  }
}

template <typename I, size_t BitsLen, size_t THREADS, size_t VEC_SIZE, size_t WF_SIZE>
__global__ void __launch_bounds__(THREADS)
    compress_matrix_vector_per_row_kernel(const size_t m, const I *row_ptr, const I *col_idx_ptr,
                                          typename CompressMatrix<I>::IdType *ids,
                                          typename CompressMatrix<I>::BitSetEleType *bits_cols, int *use_count) {
  using idtype_t = typename CompressMatrix<I>::IdType;
  using bittype_t = typename CompressMatrix<I>::BitSetEleType;

  // thread id in global
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  // thread num in global
  const int global_thread_num = gridDim.x * blockDim.x;
  // vec id in global
  const int global_vec_id = global_thread_id / VEC_SIZE;
  // vec num in global
  const int global_vec_num = global_thread_num / VEC_SIZE;
  // vec id in block
  const int block_vec_id = threadIdx.x / VEC_SIZE;
  // vec num in block
  constexpr int block_vec_num = THREADS / VEC_SIZE;
  // thread id in vec
  const int vec_thread_id = threadIdx.x % VEC_SIZE;
  // thread num in vec
  constexpr int vec_thread_num = VEC_SIZE;

  const int LDS_PER_VEC = vec_thread_num;
  __shared__ idtype_t shared_ids[LDS_PER_VEC * block_vec_num];
  __shared__ idtype_t shared_write_ids[LDS_PER_VEC * block_vec_num];
  __shared__ bittype_t shared_write_bits[LDS_PER_VEC * block_vec_num];

  idtype_t *const vec_lds_ids = &shared_ids[LDS_PER_VEC * block_vec_id];
  idtype_t *const vec_lds_write_ids = &shared_write_ids[LDS_PER_VEC * block_vec_id];
  bittype_t *const vec_lds_write_bits = &shared_write_bits[LDS_PER_VEC * block_vec_id];
  int vec_lds_count;
  int vec_last_id;

  // each vector process a row
  for (int row = global_vec_id; row < m; row += global_vec_num) {
    if (vec_thread_id == 0) {
      vec_lds_count = -1;
      vec_last_id = 0xffffffff;
    }
    const I row_idx_begin = row_ptr[row];
    const I row_idx_end = row_ptr[row + 1];

    idtype_t *vec_mem_ids = ids + row_idx_begin;
    bittype_t *vec_mem_bits = bits_cols + row_idx_begin;
    const int vector_rounds = ((row_idx_end - row_idx_begin) / vec_thread_num) +
                              (((row_idx_end - row_idx_begin) % vec_thread_num) == 0 ? 0 : 1);
    for (int round = 0; round < vector_rounds; ++round) {
      vec_lds_write_bits[vec_thread_id] = static_cast<bittype_t>(0);
      const int idx = row_idx_begin + round * vec_thread_num + vec_thread_id;
      const I col = idx < row_idx_end ? col_idx_ptr[idx] : col_idx_ptr[row_idx_end - 1];
      idtype_t my_id = col / BitsLen;
      bittype_t my_bit = idx < row_idx_end ? (0x1 << (col % BitsLen)) : static_cast<bittype_t>(0);

      vec_lds_ids[vec_thread_id] = my_id;
      __threadfence();
      int idx_offset = 0;
      if (vec_thread_id == 0) {
        if (my_id != vec_last_id) {
          vec_lds_count++;
        }
        idx_offset = 0;
      } else {
        idx_offset = (vec_lds_ids[vec_thread_id] == vec_lds_ids[vec_thread_id - 1] ? 0 : 1);
      }
      // scan
      for (int i = 1; i < VEC_SIZE; i <<= 1) {
        int tmp = __shfl_up(idx_offset, i, VEC_SIZE);
        if (vec_thread_id >= i) {
          idx_offset += tmp;
        }
      }
      atomicExch(&vec_lds_write_ids[idx_offset], my_id);
      atomicOr(&vec_lds_write_bits[idx_offset], my_bit);
      __threadfence();
      // write
      const int base_count = __shfl(vec_lds_count, 0, VEC_SIZE);
      const int lds_count = __shfl(idx_offset, VEC_SIZE - 1, VEC_SIZE);
      if (vec_thread_id <= lds_count) {
        vec_mem_ids[base_count + vec_thread_id] = vec_lds_write_ids[vec_thread_id];
        vec_mem_bits[base_count + vec_thread_id] |= vec_lds_write_bits[vec_thread_id];
      }
      vec_last_id = __shfl(my_id, VEC_SIZE - 1, VEC_SIZE);
      vec_lds_count = __shfl(base_count + lds_count, VEC_SIZE - 1, VEC_SIZE);
    }
    if (vec_thread_id == 0) {
      use_count[row] = vec_lds_count + 1;
    }
  }
}

template <typename I, size_t BitsLen, size_t THREADS, size_t WF_SIZE>
__global__ void __launch_bounds__(THREADS)
    compress_matrix_block_per_row_kernel(const size_t m, const I *row_ptr, const I *col_idx_ptr,
                                         typename CompressMatrix<I>::IdType *ids,
                                         typename CompressMatrix<I>::BitSetEleType *bits_cols, int *use_count) {
  using idtype_t = typename CompressMatrix<I>::IdType;
  using bittype_t = typename CompressMatrix<I>::BitSetEleType;

  // thread id in global
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  // thread num in global
  const int global_thread_num = gridDim.x * blockDim.x;
  // block id in global
  const int global_block_id = blockIdx.x;
  // block num in global
  const int global_block_num = gridDim.x;
  // thread id in block
  const int block_thread_id = threadIdx.x;
  // thread num in block
  constexpr int block_thread_num = THREADS;

  const int LDS_PER_BLOCK = block_thread_num;

  __shared__ idtype_t block_lds_ids[LDS_PER_BLOCK];
  __shared__ idtype_t block_lds_write_ids[LDS_PER_BLOCK];
  __shared__ bittype_t block_lds_write_bits[LDS_PER_BLOCK];
  __shared__ int block_lds_count;
  __shared__ idtype_t block_last_id;

  // each block process a row
  for (int row = global_block_id; row < m; row += global_block_num) {
    if (block_thread_id == 0) {
      block_lds_count = -1;
      block_last_id = 0xffffffff;
    }
    const I row_idx_begin = row_ptr[row];
    const I row_idx_end = row_ptr[row + 1];

    idtype_t *block_mem_ids = ids + row_idx_begin;
    bittype_t *block_mem_bits = bits_cols + row_idx_begin;
    __syncthreads();
    const int block_rounds = ((row_idx_end - row_idx_begin) / block_thread_num) +
                             (((row_idx_end - row_idx_begin) % block_thread_num) == 0 ? 0 : 1);
    for (int round = 0; round < block_rounds; ++round) {
      block_lds_write_bits[block_thread_id] = static_cast<bittype_t>(0);
      const int idx = row_idx_begin + round * block_thread_num + block_thread_id;
      idtype_t my_id;
      bittype_t my_bit;
      if (idx >= row_idx_end) {
        const I col = col_idx_ptr[row_idx_end - 1];
        my_id = col / BitsLen;
        my_bit = static_cast<bittype_t>(0);
      } else {
        const I col = col_idx_ptr[idx];
        my_id = col / BitsLen;
        my_bit = 0x1 << (col % BitsLen);
      }
      if (block_thread_id == 0) {
        if (my_id != block_last_id) {
          block_lds_count++;
        }
        block_lds_write_ids[0] = my_id;
      }
      __syncthreads();
      const int base_count = block_lds_count;
      __syncthreads();
      if (block_thread_id == 0) {
        block_lds_count = 1;
      }
      block_lds_ids[block_thread_id] = my_id;
      __syncthreads();
      const int max_id = block_lds_ids[block_thread_num - 1];
      int count = 0;
      bool flag = false;
      if (block_thread_id > 0 && block_lds_ids[block_thread_id - 1] != my_id) {
        flag = true;
        count = atomicAdd(&block_lds_count, 1);
        block_lds_write_ids[count] = my_id;
      }
      __syncthreads();
      if (my_id == max_id && flag) {
        idtype_t tmp = block_lds_write_ids[block_lds_count - 1];
        block_lds_write_ids[count] = tmp;
        block_lds_write_ids[block_lds_count - 1] = my_id;
      }
      __syncthreads();
      const int lds_count = block_lds_count;
      for (int i = 0; i < lds_count; ++i) {
        if (my_id == block_lds_write_ids[i]) {
          atomicOr(&block_lds_write_bits[i], my_bit);
        }
      }
      __syncthreads();
      // write back
      if (block_thread_id < lds_count) {
        block_mem_ids[base_count + block_thread_id] = block_lds_write_ids[block_thread_id];
        block_mem_bits[base_count + block_thread_id] |= block_lds_write_bits[block_thread_id];
      }
      if (block_thread_id == 0) {
        block_last_id = max_id;
        block_lds_count = base_count + lds_count - 1;
      }
      __syncthreads();
    }
    if (block_thread_id == 0) {
      use_count[row] = block_lds_count + 1;
    }
    // __syncthreads();
  }
}

#endif // SPGEMM_KOKKOS_SPGEMM_COMPRESS_KERNEL_HPP