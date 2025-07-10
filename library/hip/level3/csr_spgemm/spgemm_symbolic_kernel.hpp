#ifndef USTB_SPARSE_SPGEMM_SPGEMM_SYMBOLIC_KERNEL_HPP
#define USTB_SPARSE_SPGEMM_SPGEMM_SYMBOLIC_KERNEL_HPP

#include "compress_matrix.h"
#include "hash_map_insert_kernel.hpp"
#include <hip/hip_runtime.h>

template <typename I, typename BitType, size_t VEC_SIZE>
__global__ void row_nnz_kernel(const I m, const I *map_row_ptr, const BitType *map_values, const int *use_count,
                               I *row_ptr_c) {
  // thread id in global
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  // thread num in global
  const int global_thread_num = gridDim.x * blockDim.x;
  // vector id in global
  const int global_vector_id = global_thread_id / VEC_SIZE;
  // vector num in global
  const int global_vector_num = global_thread_num / VEC_SIZE;
  // thread id in vector
  const int vec_thread_id = global_thread_id % VEC_SIZE;
  // thread num in vector
  const int vec_thread_num = VEC_SIZE;

  for (int row = global_vector_id; row < m; row += global_vector_num) {
    const I idx_begin = map_row_ptr[row];
    const I idx_end = idx_begin + use_count[row];
    I non_zero_count = 0;
    for (int idx = idx_begin + vec_thread_id; idx < idx_end; idx += vec_thread_num) {
      non_zero_count += CompressMatrix<I>::pop_count(map_values[idx]);
    }
    // reduce
    for (int i = VEC_SIZE >> 1; i > 0; i >>= 1) {
      non_zero_count += __shfl_down(non_zero_count, i, VEC_SIZE);
    }
    if (vec_thread_id == 0) {
      row_ptr_c[row + 1] = non_zero_count;
    }
  }
}

template <typename I, size_t VEC_SIZE>
__global__ void chunk_nnz_compress_kernel(const size_t m, const I *row_ptr_a, const I *col_idx_ptr_a,
                                          const int *use_count, I *row_chunks_ptr) {
  using idtype_t = typename CompressMatrix<I>::IdType;
  using bittype_t = typename CompressMatrix<I>::BitSetEleType;
  // thread id in global
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  // thread num in global
  const int global_thread_num = gridDim.x * blockDim.x;
  // vector id in global
  const int global_vector_id = global_thread_id / VEC_SIZE;
  // vector num in global
  const int global_vector_num = global_thread_num / VEC_SIZE;
  // thread id in vector
  const int vec_thread_id = global_thread_id % VEC_SIZE;
  // thread num in vector
  const int vec_thread_num = VEC_SIZE;

  // each vector process a row
  for (int row_a = global_vector_id; row_a < m; row_a += global_vector_num) {
    const I row_idx_begin_a = row_ptr_a[row_a];
    const I row_idx_end_a = row_ptr_a[row_a + 1];

    I chunk_count = 0;
    for (int idx_a = row_idx_begin_a + vec_thread_id; idx_a < row_idx_end_a; idx_a += vec_thread_num) {
      const I row_b = col_idx_ptr_a[idx_a];
      chunk_count += use_count[row_b];
    }
    // reduce
    for (int i = VEC_SIZE >> 1; i > 0; i >>= 1) {
      chunk_count += __shfl_down(chunk_count, i, VEC_SIZE);
    }
    if (vec_thread_id == 0) {
      row_chunks_ptr[row_a + 1] = chunk_count;
    }
  }
}

template <typename I, size_t VEC_SIZE>
__global__ void chunk_nnz_kernel(const size_t m, const I *row_ptr_a, const I *col_idx_ptr_a, const I *row_ptr_b,
                                 I *row_chunks_ptr) {
  // thread id in global
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  // thread num in global
  const int global_thread_num = gridDim.x * blockDim.x;
  // vector id in global
  const int global_vector_id = global_thread_id / VEC_SIZE;
  // vector num in global
  const int global_vector_num = global_thread_num / VEC_SIZE;
  // thread id in vector
  const int vec_thread_id = global_thread_id % VEC_SIZE;
  // thread num in vector
  const int vec_thread_num = VEC_SIZE;

  // each vector process a row
  for (int row_a = global_vector_id; row_a < m; row_a += global_vector_num) {
    const I row_idx_begin_a = row_ptr_a[row_a];
    const I row_idx_end_a = row_ptr_a[row_a + 1];

    I chunk_count = 0;
    for (int idx_a = row_idx_begin_a + vec_thread_id; idx_a < row_idx_end_a; idx_a += vec_thread_num) {
      const I row_b = col_idx_ptr_a[idx_a];
      const I nnz_b = row_ptr_b[row_b + 1] - row_ptr_b[row_b];
      chunk_count += nnz_b;
    }
    // reduce
    for (int i = VEC_SIZE >> 1; i > 0; i >>= 1) {
      chunk_count += __shfl_down(chunk_count, i, VEC_SIZE);
    }
    if (vec_thread_id == 0) {
      row_chunks_ptr[row_a + 1] = chunk_count;
    }
  }
}

#if 0
template <typename T, typename I, size_t N, size_t THREADS, size_t VEC_SIZE>
__global__ void __launch_bounds__(THREADS)
    spgemm_symbolic_vector_per_row_one_pass_kernel(const size_t m, const I *row_ptr_a, const I *col_idx_ptr_a,
                                                   const I *row_ptr_b,
                                                   const typename CompressMatrix<I>::IdType *ids_ptr_b,
                                                   const typename CompressMatrix<I>::BitSetEleType *col_idx_ptr_b,
                                                   const int *use_count, I *row_ptr_c) {
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
  const int vec_thread_id = global_thread_id % VEC_SIZE;
  // thread num in vec
  constexpr int vec_thread_num = VEC_SIZE;
  // hashmap in LDS
  constexpr size_t LDS_MAP_CAP = VEC_SIZE * N;
  constexpr size_t LDS_HASH_MASK = LDS_MAP_CAP - 1;
  constexpr int BLOCK_LDS_SIZE = LDS_MAP_CAP * block_vec_num;

  if (global_vec_id >= m) {
    return;
  }
  const int row_a = global_vec_id;

  __shared__ bittype_t shared_values[BLOCK_LDS_SIZE];
  bittype_t *vec_lds_values = shared_values + LDS_MAP_CAP * block_vec_id;
  // init lds array
  for (int i = vec_thread_id; i < LDS_MAP_CAP; i += vec_thread_num) {
    vec_lds_values[i] = static_cast<bittype_t>(0);
  }
  const I row_idx_begin = row_ptr_a[row_a];
  const I row_idx_end = row_ptr_a[row_a + 1];
  if(row_idx_end - row_idx_begin <= static_cast<I>(1)) return;

  for (int idx_a = row_idx_begin; idx_a < row_idx_end; ++idx_a) {
    const I col_a = col_idx_ptr_a[idx_a];

    const I row_b = col_a;
    const I idx_begin_b = row_ptr_b[row_b];
    const idtype_t *vec_mem_id_b = ids_ptr_b + idx_begin_b;
    const bittype_t *vec_mem_col_b = col_idx_ptr_b + idx_begin_b;
    const int b_count = use_count[row_b];
    for (int idx_b = vec_thread_id; idx_b < b_count; idx_b += vec_thread_num) {
      atomicOr(vec_lds_values + vec_mem_id_b[idx_b], vec_mem_col_b[idx_b]);
    }
  }
  I nnz_count = 0;
  for (int idx = vec_thread_id; idx < LDS_MAP_CAP; idx += vec_thread_num) {
    nnz_count += CompressMatrix<I>::pop_count(vec_lds_values[idx]);
  }
  for (int i = VEC_SIZE >> 1; i > 0; i >>= 1) {
    nnz_count += __shfl_down(nnz_count, i, VEC_SIZE);
  }
  if (vec_thread_id == 0) {
    row_ptr_c[row_a + 1] = nnz_count;
  }
}
#endif

#if 0
template <typename T, typename I, size_t N, size_t THREADS>
__global__ void __launch_bounds__(THREADS)
    spgemm_symbolic_block_per_row_one_pass_kernel(const size_t m, const I *row_ptr_a, const I *col_idx_ptr_a,
                                                  const I *row_ptr_b,
                                                  const typename CompressMatrix<I>::IdType *ids_ptr_b,
                                                  const typename CompressMatrix<I>::BitSetEleType *col_idx_ptr_b,
                                                  const int *use_count, I *row_ptr_c) {
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
  // wavefront id in block
  const int block_wf_id = block_thread_id / __WF_SIZE__;
  // wavefront num in block
  constexpr int block_wf_num = THREADS / __WF_SIZE__;
  // thread id in wavefront
  const int wf_thread_id = block_thread_id % __WF_SIZE__;
  // thread num in wavefront
  const int wf_thread_num = __WF_SIZE__;
  // array in LDS
  constexpr size_t LDS_MAP_CAP = THREADS * N;
  constexpr int BLOCK_LDS_SIZE = LDS_MAP_CAP;

  if (global_block_id >= m) {
    return;
  }
  const int row_a = global_block_id;

  __shared__ bittype_t shared_values[BLOCK_LDS_SIZE];
  __shared__ I shared_nnz_count;

  bittype_t *block_lds_values = shared_values;
  // init lds hashmap
  for (int i = block_thread_id; i < BLOCK_LDS_SIZE; i += block_thread_num) {
    shared_values[i] = static_cast<bittype_t>(0);
  }
  if (block_thread_id == 0) {
    shared_nnz_count = static_cast<I>(0);
  }
  const I row_idx_begin = row_ptr_a[row_a];
  const I row_idx_end = row_ptr_a[row_a + 1];
  if(row_idx_end - row_idx_begin <= static_cast<I>(1)) return;
  __syncthreads();

  for (int idx_a = row_idx_begin + block_wf_id; idx_a < row_idx_end; idx_a += block_wf_num) {
    const I col_a = col_idx_ptr_a[idx_a];

    const I row_b = col_a;
    const I idx_begin_b = row_ptr_b[row_b];
    const idtype_t *block_mem_id_b = ids_ptr_b + idx_begin_b;
    const bittype_t *block_mem_col_b = col_idx_ptr_b + idx_begin_b;
    const int b_count = use_count[row_b];
    for (int idx_b = wf_thread_id; idx_b < b_count; idx_b += wf_thread_num) {
      atomicOr(block_lds_values + block_mem_id_b[idx_b], block_mem_col_b[idx_b]);
    }
  }
  __syncthreads();
  I nnz_count = 0;
  for (int idx = block_thread_id; idx < LDS_MAP_CAP; idx += block_thread_num) {
    nnz_count += CompressMatrix<I>::pop_count(block_lds_values[idx]);
  }
  for (int i = __WF_SIZE__ >> 1; i > 0; i >>= 1) {
    nnz_count += __shfl_down(nnz_count, i, __WF_SIZE__);
  }
  if (wf_thread_id == 0) {
    atomicAdd(&shared_nnz_count, nnz_count);
  }
  __syncthreads();
  if (block_thread_id == 0) {
    row_ptr_c[row_a + 1] = shared_nnz_count;
  }
}
#endif

template <typename T, typename I, size_t N, size_t THREADS, size_t VEC_SIZE>
__global__ void __launch_bounds__(THREADS) spgemm_symbolic_open_address_hashmap_vector_per_row_one_pass_kernel(
    const size_t m, const I *row_ptr_a, const I *col_idx_ptr_a, const I *row_ptr_b,
    const typename CompressMatrix<I>::IdType *ids_ptr_b, const typename CompressMatrix<I>::BitSetEleType *col_idx_ptr_b,
    const int *use_count, I *row_ptr_c) {
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
  const int vec_thread_id = global_thread_id % VEC_SIZE;
  // thread num in vec
  constexpr int vec_thread_num = VEC_SIZE;
  // hashmap in LDS
  constexpr size_t LDS_MAP_CAP = VEC_SIZE * N;
  constexpr size_t LDS_HASH_MASK = LDS_MAP_CAP - 1;
  constexpr int BLOCK_LDS_SIZE = LDS_MAP_CAP * block_vec_num;

  if (global_vec_id >= m) {
    return;
  }
  const int row_a = global_vec_id;

  __shared__ idtype_t shared_ids[BLOCK_LDS_SIZE];
  __shared__ bittype_t shared_values[BLOCK_LDS_SIZE];
  idtype_t *vec_lds_ids = shared_ids + LDS_MAP_CAP * block_vec_id;
  bittype_t *vec_lds_values = shared_values + LDS_MAP_CAP * block_vec_id;

  // init lds hashmap
  for (int i = vec_thread_id; i < LDS_MAP_CAP; i += vec_thread_num) {
    vec_lds_ids[i] = CompressMatrix<I>::NullValue;
    vec_lds_values[i] = static_cast<bittype_t>(0);
  }
  const I row_idx_begin = row_ptr_a[row_a];
  const I row_idx_end = row_ptr_a[row_a + 1];
  for (int idx_a = row_idx_begin; idx_a < row_idx_end; ++idx_a) {
    const I col_a = col_idx_ptr_a[idx_a];

    const I row_b = col_a;
    const I idx_begin_b = row_ptr_b[row_b];
    const idtype_t *vec_mem_id_b = ids_ptr_b + idx_begin_b;
    const bittype_t *vec_mem_col_b = col_idx_ptr_b + idx_begin_b;
    const int b_count = use_count[row_b];
    for (int i = vec_thread_id; i < b_count; i += vec_thread_num) {
      // insert to lds hashmap
      open_address_hashmap_vector_insert_or<bittype_t, idtype_t, CompressMatrix<I>::NullValue, LDS_HASH_MASK>(
          vec_lds_ids, vec_lds_values, vec_mem_id_b[i], vec_mem_col_b[i]);
    }
  }
  I local_nnz_count = static_cast<I>(0);
  for (int i = vec_thread_id; i < LDS_MAP_CAP; i += vec_thread_num) {
    local_nnz_count += CompressMatrix<I>::pop_count(vec_lds_values[i]);
  }
  // sum
  for (int i = VEC_SIZE >> 1; i > 0; i >>= 1) {
    local_nnz_count += __shfl_down(local_nnz_count, i, VEC_SIZE);
  }
  if (vec_thread_id == 0) {
    row_ptr_c[row_a + 1] = local_nnz_count;
  }
}

template <typename T, typename I, size_t N, size_t THREADS>
__global__ void __launch_bounds__(THREADS) spgemm_symbolic_open_address_hashmap_block_per_row_one_pass_kernel(
    const size_t m, const I *row_ptr_a, const I *col_idx_ptr_a, const I *row_ptr_b,
    const typename CompressMatrix<I>::IdType *ids_ptr_b, const typename CompressMatrix<I>::BitSetEleType *col_idx_ptr_b,
    const int *use_count, I *row_ptr_c) {
  using idtype_t = typename CompressMatrix<I>::IdType;
  using bittype_t = typename CompressMatrix<I>::BitSetEleType;
  // block id in global
  const int global_block_id = blockIdx.x;
  // block num in global
  const int global_block_num = gridDim.x;
  // thread id in block
  const int block_thread_id = threadIdx.x;
  // thread num in block
  constexpr int block_thread_num = THREADS;
  // wavefront id in block
  const int block_wf_id = block_thread_id / __WF_SIZE__;
  // wavefront num in block
  constexpr int block_wf_num = THREADS / __WF_SIZE__;
  // thread id in wavefront
  const int wf_thread_id = block_thread_id % __WF_SIZE__;
  // thread num in wavefront
  const int wf_thread_num = __WF_SIZE__;
  // hashmap in LDS
  constexpr size_t LDS_MAP_CAP = THREADS * N;
  constexpr size_t LDS_HASH_MASK = LDS_MAP_CAP - 1;
  constexpr int BLOCK_LDS_SIZE = LDS_MAP_CAP;

  __shared__ idtype_t block_lds_ids[BLOCK_LDS_SIZE];
  __shared__ bittype_t block_lds_values[BLOCK_LDS_SIZE];
  __shared__ I shared_nnz_count;

  if (global_block_id >= m) {
    return;
  }
  const int row_a = global_block_id;
  // init lds hashmap
  for (int i = block_thread_id; i < BLOCK_LDS_SIZE; i += block_thread_num) {
    block_lds_ids[i] = CompressMatrix<I>::NullValue;
    block_lds_values[i] = static_cast<bittype_t>(0);
  }
  if (block_thread_id == 0) {
    shared_nnz_count = static_cast<I>(0);
  }
  const I row_idx_begin = row_ptr_a[row_a];
  const I row_idx_end = row_ptr_a[row_a + 1];
  __syncthreads();
  for (int idx_a = row_idx_begin + block_wf_id; idx_a < row_idx_end; idx_a += block_wf_num) {
    const I col_a = col_idx_ptr_a[idx_a];

    const I row_b = col_a;
    const I idx_begin_b = row_ptr_b[row_b];
    const idtype_t *block_mem_id_b = ids_ptr_b + idx_begin_b;
    const bittype_t *block_mem_col_b = col_idx_ptr_b + idx_begin_b;
    const int b_count = use_count[row_b];
    for (int i = wf_thread_id; i < b_count; i += wf_thread_num) {
      // insert to lds hashmap
      open_address_hashmap_vector_insert_or<bittype_t, idtype_t, CompressMatrix<I>::NullValue, LDS_HASH_MASK>(
          block_lds_ids, block_lds_values, block_mem_id_b[i], block_mem_col_b[i]);
    }
  }
  __syncthreads();
  I local_nnz_count = static_cast<I>(0);
  for (int i = block_thread_id; i < BLOCK_LDS_SIZE; i += block_thread_num) {
    local_nnz_count += CompressMatrix<I>::pop_count(block_lds_values[i]);
  }
  __syncthreads();
  // reduce
  for (int i = __WF_SIZE__ >> 1; i > 0; i >>= 1) {
    local_nnz_count += __shfl_down(local_nnz_count, i, __WF_SIZE__);
  }
  if (wf_thread_id == 0) {
    atomicAdd(&shared_nnz_count, local_nnz_count);
  }
  __syncthreads();
  if (block_thread_id == 0) {
    row_ptr_c[row_a + 1] = shared_nnz_count;
  }
}

template <typename T, typename I, size_t N, size_t THREADS, size_t VEC_SIZE, size_t BT_SIZE>
__global__ void __launch_bounds__(THREADS)
    spgemm_symbolic_open_address_hashmap_vector_per_row_one_pass_kernel(const size_t m, const I *row_ptr_a,
                                                                        const I *col_idx_ptr_a, const I *row_ptr_b,
                                                                        const I *col_idx_ptr_b, I *row_ptr_c) {
  // thread id in global
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  // thread num in global
  const int global_thread_num = gridDim.x * blockDim.x;
  // vec id in global
  const int global_vec_id = global_thread_id / VEC_SIZE;
  // vec num in global
  const int global_vec_num = global_thread_num / VEC_SIZE;
  // block id in global
  const int global_block_id = blockIdx.x;
  // vec id in block
  const int block_vec_id = threadIdx.x / VEC_SIZE;
  // vec num in block
  constexpr int block_vec_num = THREADS / VEC_SIZE;
  // thread id in vec
  const int vec_thread_id = threadIdx.x % VEC_SIZE;
  // thread num in vec
  constexpr int vec_thread_num = VEC_SIZE;
  // bt id in vec
  const int vec_bt_id = vec_thread_id / BT_SIZE;
  // bt num in vec
  constexpr int vec_bt_num = VEC_SIZE / BT_SIZE;
  // thread id in bt
  const int bt_thread_id = threadIdx.x % BT_SIZE;
  // thread num in bt
  constexpr int bt_thread_num = BT_SIZE;

  // hashmap in LDS
  constexpr size_t LDS_MAP_CAP = VEC_SIZE * N;
  constexpr size_t LDS_HASH_MASK = LDS_MAP_CAP - 1;
  constexpr int BLOCK_LDS_SIZE = LDS_MAP_CAP * block_vec_num;
  constexpr I HASH_SET_NULL_VALUE = 0xffffffff;

  if (global_vec_id >= m) {
    return;
  }
  const int row_a = global_vec_id;
  __shared__ I shared_ids[BLOCK_LDS_SIZE];
  I *vec_lds_ids = shared_ids + LDS_MAP_CAP * block_vec_id;

  I local_nnz_count = static_cast<I>(0);
  // init lds hashmap
  for (int i = vec_thread_id; i < LDS_MAP_CAP; i += vec_thread_num) {
    vec_lds_ids[i] = HASH_SET_NULL_VALUE;
  }
  const I row_idx_begin_a = row_ptr_a[row_a];
  const I row_idx_end_a = row_ptr_a[row_a + 1];
  for (int idx_a = row_idx_begin_a + vec_bt_id; idx_a < row_idx_end_a; idx_a += vec_bt_num) {
    const I col_a = col_idx_ptr_a[idx_a];

    const I row_b = col_a;
    const I row_idx_begin_b = row_ptr_b[row_b];
    const I row_idx_end_b = row_ptr_b[row_b + 1];
    for (int idx_b = row_idx_begin_b + bt_thread_id; idx_b < row_idx_end_b; idx_b += bt_thread_num) {
      local_nnz_count += open_address_hashset_vector_insert<T, I, HASH_SET_NULL_VALUE, LDS_HASH_MASK>(
          vec_lds_ids, col_idx_ptr_b[idx_b]);
    }
  }
  // reduce
  for (int i = VEC_SIZE >> 1; i > 0; i >>= 1) {
    local_nnz_count += __shfl_down(local_nnz_count, i, VEC_SIZE);
  }
  if (vec_thread_id == 0) {
    row_ptr_c[row_a + 1] = local_nnz_count;
  }
}

template <typename T, typename I, size_t N, size_t THREADS>
__global__ void __launch_bounds__(THREADS)
    spgemm_symbolic_open_address_hashmap_block_per_row_one_pass_kernel(const size_t m, const I *row_ptr_a,
                                                                       const I *col_idx_ptr_a, const I *row_ptr_b,
                                                                       const I *col_idx_ptr_b, I *row_ptr_c) {
  // block id in global
  const int global_block_id = blockIdx.x;
  // block num in global
  const int global_block_num = gridDim.x;
  // thread id in block
  const int block_thread_id = threadIdx.x;
  // thread num in block
  constexpr int block_thread_num = THREADS;
  // wavefront id in block
  const int block_wf_id = block_thread_id / __WF_SIZE__;
  // wavefront num in block
  constexpr int block_wf_num = THREADS / __WF_SIZE__;
  // thread id in wavefront
  const int wf_thread_id = block_thread_id % __WF_SIZE__;
  // thread num in wavefront
  const int wf_thread_num = __WF_SIZE__;
  // hashmap in LDS
  constexpr size_t LDS_MAP_CAP = THREADS * N;
  constexpr size_t LDS_HASH_MASK = LDS_MAP_CAP - 1;
  constexpr int BLOCK_LDS_SIZE = LDS_MAP_CAP;
  constexpr I HASH_SET_NULL_VALUE = 0xffffffff;

  if (global_block_id >= m) {
    return;
  }
  const int row_a = global_block_id;

  __shared__ I block_lds_ids[BLOCK_LDS_SIZE];
  __shared__ I shared_nnz_count;

  // init lds hashmap
  for (int i = block_thread_id; i < BLOCK_LDS_SIZE; i += block_thread_num) {
    block_lds_ids[i] = HASH_SET_NULL_VALUE;
  }
  if (block_thread_id == 0) {
    shared_nnz_count = static_cast<I>(0);
  }

  const I row_idx_begin_a = row_ptr_a[row_a];
  const I row_idx_end_a = row_ptr_a[row_a + 1];
  I local_nnz_count = static_cast<I>(0);
  __syncthreads();
  for (int idx_a = row_idx_begin_a + block_wf_id; idx_a < row_idx_end_a; idx_a += block_wf_num) {
    const I col_a = col_idx_ptr_a[idx_a];
    const I row_b = col_a;
    const I row_idx_begin_b = row_ptr_b[row_b];
    const I row_idx_end_b = row_ptr_b[row_b + 1];

    for (int idx_b = row_idx_begin_b + wf_thread_id; idx_b < row_idx_end_b; idx_b += wf_thread_num) {
      // insert to lds hashmap
      local_nnz_count += open_address_hashset_vector_insert<T, I, HASH_SET_NULL_VALUE, LDS_HASH_MASK>(
          block_lds_ids, col_idx_ptr_b[idx_b]);
    }
  }
  // block sum
  for (int i = __WF_SIZE__ >> 1; i > 0; i >>= 1) {
    local_nnz_count += __shfl_down(local_nnz_count, i, __WF_SIZE__);
  }
  __syncthreads();
  if (wf_thread_id == 0) {
    atomicAdd(&shared_nnz_count, local_nnz_count);
  }
  __syncthreads();
  if (block_thread_id == 0) {
    row_ptr_c[row_a + 1] = shared_nnz_count;
  }
}

// without hash
template <typename T, typename I, size_t N, size_t THREADS, size_t VEC_SIZE>
__global__ void __launch_bounds__(THREADS)
    spgemm_symbolic_bit_map_vector_per_row_one_pass_kernel(const size_t m, const I *row_ptr_a, const I *col_idx_ptr_a,
                                                           const I *row_ptr_b, const I *col_idx_ptr_b, I *row_ptr_c) {
  // thread id in global
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  // thread num in global
  const int global_thread_num = gridDim.x * blockDim.x;
  // vec id in global
  const int global_vec_id = global_thread_id / VEC_SIZE;
  // vec num in global
  const int global_vec_num = global_thread_num / VEC_SIZE;
  // block id in global
  const int global_block_id = blockIdx.x;
  // thread id in block
  const int block_thread_id = threadIdx.x;
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

  const int row_a = global_vec_id;
  if (row_a >= m) {
    return;
  }

  extern __shared__ unsigned long long int shared_bit_map[];

  constexpr size_t BitsLen = sizeof(unsigned long long int) * 8;
  const size_t VEC_LDS_NUM = m / BitsLen + (m % BitsLen == 0 ? 0 : 1);
  unsigned long long int *vec_lds_ids = shared_bit_map + VEC_LDS_NUM * block_vec_id;

  // init lds hashmap
  for (int i = vec_thread_id; i < VEC_LDS_NUM; i += vec_thread_num) {
    vec_lds_ids[i] = static_cast<unsigned long long>(0);
  }
  const I row_idx_begin_a = row_ptr_a[row_a];
  const I row_idx_end_a = row_ptr_a[row_a + 1];

  for (int idx_a = row_idx_begin_a; idx_a < row_idx_end_a; ++idx_a) {
    const I col_a = col_idx_ptr_a[idx_a];
    const I row_b = col_a;
    const I row_idx_begin_b = row_ptr_b[row_b];
    const I row_idx_end_b = row_ptr_b[row_b + 1];

    for (int idx_b = row_idx_begin_b + vec_thread_id; idx_b < row_idx_end_b; idx_b += vec_thread_num) {
      I col_b = col_idx_ptr_b[idx_b];
      unsigned long long mask = static_cast<unsigned long long>(0x1) << (col_b % BitsLen);
      atomicOr(vec_lds_ids + col_b / BitsLen, mask);
    }
  }

  int local_use_count = static_cast<int>(0);
  for (int i = vec_thread_id; i < VEC_LDS_NUM; i += vec_thread_num) {
    local_use_count += __popcll(vec_lds_ids[i]);
  }

  for (int i = VEC_SIZE >> 1; i > 0; i >>= 1) {
    local_use_count += __shfl_down(local_use_count, i, VEC_SIZE);
  }

  if (vec_thread_id == 0) {
    row_ptr_c[row_a + 1] = local_use_count;
  }
}

template <typename T, typename I, size_t CHUNK_SIZE, size_t THREADS, size_t VEC_SIZE>
__global__ void __launch_bounds__(THREADS)
    spgemm_symbolic_block_per_row_bp_kernel(const size_t m, const I *row_ptr_a, const I *col_idx_ptr_a,
                                            const I *row_ptr_b, const I *col_idx_ptr_b, I *row_ptr_c, const I *bp,
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
  // thread id in wf
  const int wf_thread_id = threadIdx.x % __WF_SIZE__;
  // thread num in vec
  constexpr int vec_thread_num = VEC_SIZE;
  // chunks
  __shared__ bool table[CHUNK_SIZE];
  __shared__ I shared_nnz_count;

  // block processes a row
  if (global_block_id >= m) {
    return;
  }
  if (block_thread_id == 0) {
    shared_nnz_count = static_cast<I>(0);
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
    }
    __syncthreads();

    for (I idx_a = row_idx_begin + block_vec_id; idx_a < row_idx_end; idx_a += block_vec_num) {
      const I col_a = col_idx_ptr_a[idx_a];
      const I row_b = col_a;

      I idx_begin_b = bp[row_b * BP_IDX_PER_ROW + bp_idx];
      I idx_end_b = bp[row_b * BP_IDX_PER_ROW + bp_idx + 1];

      for (I idx_b = idx_begin_b + vec_thread_id; idx_b < idx_end_b; idx_b += vec_thread_num) {
        const I col_b = col_idx_ptr_b[idx_b];
        table[col_b - chunk_begin] = 1;
      }
    }
    __syncthreads();
    I local_nnz_count = static_cast<I>(0);
    for (int i = block_thread_id; i < CHUNK_SIZE; i += block_thread_num) {
      if (table[i]) {
        local_nnz_count++;
      }
    }
    for (int i = __WF_SIZE__ >> 1; i > 0; i >>= 1) {
      local_nnz_count += __shfl_down(local_nnz_count, i);
    }
    if (wf_thread_id == 0) {
      atomicAdd(&shared_nnz_count, local_nnz_count);
    }
    __syncthreads();
  }
  if (block_thread_id == 0) {
    row_ptr_c[row_a + 1] = shared_nnz_count;
  }
}

#endif // USTB_SPARSE_SPGEMM_SPGEMM_SYMBOLIC_KERNEL_HPP
