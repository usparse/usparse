#ifndef SPGEMM_KOKKOS_SPGEMM_SYMBOLIC_KERNEL_HPP
#define SPGEMM_KOKKOS_SPGEMM_SYMBOLIC_KERNEL_HPP

#include "building_config.h"
#include "compress_matrix.h"
#include "hash_map.h"
#include "hash_map_insert.hpp"
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
  const int vector_thread_id = global_thread_id % VEC_SIZE;
  // thread num in vector
  const int vector_thread_num = VEC_SIZE;

  for (int row = global_vector_id; row < m; row += global_vector_num) {
    const I idx_begin = map_row_ptr[row];
    const I idx_end = idx_begin + use_count[row];
    I non_zero_count = 0;
    for (int idx = idx_begin + vector_thread_id; idx < idx_end; idx += vector_thread_num) {
      non_zero_count += __popcll(map_values[idx]);
    }
    // reduce
    for (int i = VEC_SIZE >> 1; i > 0; i >>= 1) {
      non_zero_count += __shfl_down(non_zero_count, i, VEC_SIZE);
    }
    if (vector_thread_id == 0) {
      row_ptr_c[row + 1] = non_zero_count;
    }
  }
}

template <typename I, size_t VEC_SIZE>
__global__ void trunk_nums_kernel(const size_t m, const I *row_ptr_a, const I *col_idx_ptr_a, const int *use_count,
                                  I *row_trunks_ptr) {
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
  const int vector_thread_id = global_thread_id % VEC_SIZE;
  // thread num in vector
  const int vector_thread_num = VEC_SIZE;

  // each vector process a row
  for (int row_a = global_vector_id; row_a < m; row_a += global_vector_num) {
    const I row_idx_begin_a = row_ptr_a[row_a];
    const I row_idx_end_a = row_ptr_a[row_a + 1];

    I trunk_count = 0;
    for (int idx_a = row_idx_begin_a + vector_thread_id; idx_a < row_idx_end_a; idx_a += vector_thread_num) {
      const I row_b = col_idx_ptr_a[idx_a];
      trunk_count += use_count[row_b];
    }
    // reduce
    for (int i = VEC_SIZE >> 1; i > 0; i >>= 1) {
      trunk_count += __shfl_down(trunk_count, i, VEC_SIZE);
    }
    if (vector_thread_id == 0) {
      row_trunks_ptr[row_a + 1] = trunk_count;
    }
  }
}

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
  // hashmap in LDS
  constexpr size_t LDS_MAP_CAP = THREADS * N;

  // hashmap in LDS
  constexpr int BLOCK_LDS_SIZE = LDS_MAP_CAP;

  __shared__ bittype_t shared_values[BLOCK_LDS_SIZE];
  __shared__ I shared_nnz_count[block_wf_num];

  idtype_t *block_lds_ids = shared_ids;
  bittype_t *block_lds_values = shared_values;
  // init lds hashmap
  for (int i = block_thread_id; i < BLOCK_LDS_SIZE; i += block_thread_num) {
    shared_values[i] = 0;
  }
  for (int i = block_thread_id; i < block_wf_num; i += block_thread_num) {
    shared_nnz_count[i] = 0;
  }
  __syncthreads();

  int row_a = global_block_id;

  if (row_a >= m) {
    return;
  }

  const I row_idx_begin = row_ptr_a[row_a];
  const I row_idx_end = row_ptr_a[row_a + 1];

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
    nnz_count += __popcll(block_lds_values[idx]);
  }
  __syncthreads();
  // wf reduce
  for (int i = __WF_SIZE__ >> 1; i > 0; i >>= 1) {
    nnz_count += __shfl_down(nnz_count, i, __WF_SIZE__);
  }
  if (wf_thread_id == 0) {
    shared_nnz_count[block_wf_id] = nnz_count;
  }
  __syncthreads();
  // block reduce
  if (block_wf_id == 0) {
    nnz_count = 0;
    for (int i = wf_thread_id; i < block_wf_num; i += wf_thread_num) {
      nnz_count += shared_nnz_count[i];
    }
    // wf reduce
    for (int i = __WF_SIZE__ >> 1; i > 0; i >>= 1) {
      nnz_count += __shfl_down(nnz_count, i, __WF_SIZE__);
    }
    if (wf_thread_id == 0) {
      row_ptr_c[row_a + 1] = nnz_count;
    }
  }
}

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
  // hashmap in LDS
  constexpr size_t LDS_MAP_CAP = VEC_SIZE * N;
  // hash mask
  constexpr size_t LDS_HASH_MASK = LDS_MAP_CAP - 1;

  constexpr int BLOCK_LDS_SIZE = LDS_MAP_CAP * block_vec_num;

  __shared__ bittype_t shared_values[BLOCK_LDS_SIZE];

  bittype_t *vec_lds_values = shared_values + LDS_MAP_CAP * block_vec_id;
  // init lds array
  for (int i = block_thread_id; i < BLOCK_LDS_SIZE; i += block_thread_num) {
    shared_values[i] = 0;
  }
  __syncthreads();

  int row_a = global_vec_id;

  if (row_a >= m) {
    return;
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
    for (int idx_b = vec_thread_id; idx_b < b_count; idx_b += vec_thread_num) {
      atomicOr(vec_lds_values + vec_mem_id_b[idx_b], vec_mem_col_b[idx_b]);
    }
  }
  I nnz_count = 0;
  for (int idx = vec_thread_id; idx < LDS_MAP_CAP; idx += vec_thread_num) {
    nnz_count += __popcll(vec_lds_values[idx]);
  }
  // vec reduce
  for (int i = VEC_SIZE >> 1; i > 0; i >>= 1) {
    nnz_count += __shfl_down(nnz_count, i, VEC_SIZE);
  }
  if (vec_thread_id == 0) {
    row_ptr_c[row_a + 1] = nnz_count;
  }
}

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
  // hashmap in LDS
  constexpr size_t LDS_MAP_CAP = VEC_SIZE * N;
  // hash mask
  constexpr size_t LDS_HASH_MASK = LDS_MAP_CAP - 1;

  constexpr int BLOCK_LDS_SIZE = LDS_MAP_CAP * block_vec_num;

  const int row_a = global_vec_id;
  if (row_a >= m) {
    return;
  }

  __shared__ idtype_t shared_ids[BLOCK_LDS_SIZE];
  __shared__ bittype_t shared_values[BLOCK_LDS_SIZE];

  idtype_t *vec_lds_ids = shared_ids + LDS_MAP_CAP * block_vec_id;
  bittype_t *vec_lds_values = shared_values + LDS_MAP_CAP * block_vec_id;

  // init lds hashmap
  for (int i = vec_thread_id; i < LDS_MAP_CAP; i += vec_thread_num) {
    vec_lds_ids[i] = HashMap<bittype_t, idtype_t>::NullValue;
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
      open_address_hashmap_vector_insert_or<bittype_t, idtype_t, LDS_HASH_MASK>(vec_lds_ids, vec_lds_values,
                                                                                vec_mem_id_b[i], vec_mem_col_b[i]);
    }
  }
  I local_use_count = static_cast<I>(0);
  for (int i = vec_thread_id; i < LDS_MAP_CAP; i += vec_thread_num) {
    local_use_count += __popcll(vec_lds_values[i]);
  }
  // sum
  for (int i = VEC_SIZE >> 1; i > 0; i >>= 1) {
    local_use_count += __shfl_down(local_use_count, i, VEC_SIZE);
  }
  if (vec_thread_id == 0) {
    row_ptr_c[row_a + 1] = local_use_count;
  }
}

template <typename T, typename I, size_t N, size_t THREADS>
__global__ void __launch_bounds__(THREADS) spgemm_symbolic_open_address_hashmap_block_per_row_one_pass_kernel(
    const size_t m, const I *row_ptr_a, const I *col_idx_ptr_a, const I *row_ptr_b,
    const typename CompressMatrix<I>::IdType *ids_ptr_b, const typename CompressMatrix<I>::BitSetEleType *col_idx_ptr_b,
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
  // hashmap in LDS
  constexpr size_t LDS_MAP_CAP = THREADS * N;
  // hash mask
  constexpr size_t LDS_HASH_MASK = LDS_MAP_CAP - 1;

  constexpr int BLOCK_LDS_SIZE = LDS_MAP_CAP;

  __shared__ idtype_t shared_ids[BLOCK_LDS_SIZE];
  __shared__ bittype_t shared_values[BLOCK_LDS_SIZE];
  __shared__ I shared_sum[THREADS / __WF_SIZE__];

  idtype_t *block_lds_ids = shared_ids;
  bittype_t *block_lds_values = shared_values;

  // init lds hashmap
  for (int i = block_thread_id; i < BLOCK_LDS_SIZE; i += block_thread_num) {
    shared_ids[i] = HashMap<bittype_t, idtype_t>::NullValue;
    shared_values[i] = static_cast<bittype_t>(0);
  }
  for (int i = block_thread_id; i < THREADS / __WF_SIZE__; i += block_thread_num) {
    shared_sum[i] = static_cast<I>(0);
  }
  __syncthreads();

  const int row_a = global_block_id;
  if (row_a >= m) {
    return;
  }
  const I row_idx_begin = row_ptr_a[row_a];
  const I row_idx_end = row_ptr_a[row_a + 1];

  for (int idx_a = row_idx_begin + block_wf_id; idx_a < row_idx_end; idx_a += block_wf_num) {
    const I col_a = col_idx_ptr_a[idx_a];
    const I row_b = col_a;
    const I idx_begin_b = row_ptr_b[row_b];
    const idtype_t *block_mem_id_b = ids_ptr_b + idx_begin_b;
    const bittype_t *block_mem_col_b = col_idx_ptr_b + idx_begin_b;
    const int b_count = use_count[row_b];
    for (int i = wf_thread_id; i < b_count; i += wf_thread_num) {
      // insert to lds hashmap
      open_address_hashmap_vector_insert_or<bittype_t, idtype_t, LDS_HASH_MASK>(block_lds_ids, block_lds_values,
                                                                                block_mem_id_b[i], block_mem_col_b[i]);
    }
  }
  __syncthreads();
  I local_use_count = static_cast<I>(0);
  for (int i = block_thread_id; i < BLOCK_LDS_SIZE; i += block_thread_num) {
    local_use_count += __popcll(block_lds_values[i]);
  }
  __syncthreads();
  // block sum
  for (int i = __WF_SIZE__ >> 1; i > 0; i >>= 1) {
    local_use_count += __shfl_down(local_use_count, i, __WF_SIZE__);
  }
  if (wf_thread_id == 0) {
    shared_sum[block_wf_id] += local_use_count;
  }
  __syncthreads();
  if (block_wf_id == 0) {
    local_use_count = static_cast<I>(0);
    for (int i = wf_thread_id; i < THREADS / __WF_SIZE__; i += wf_thread_num) {
      local_use_count += shared_sum[i];
    }
    for (int i = __WF_SIZE__ >> 1; i > 0; i >>= 1) {
      local_use_count += __shfl_down(local_use_count, i, __WF_SIZE__);
    }
    if (wf_thread_id == 0) {
      row_ptr_c[row_a + 1] = local_use_count;
    }
  }
}

template <typename T, typename I, size_t N, size_t LDS_MAP_CAP, size_t LDS_HASH_MASK, size_t THREADS>
__global__ void __launch_bounds__(THREADS) spgemm_symbolic_linear_chain_hashmap_block_per_row_kernel(
    const size_t m, const I *row_ptr_a, const I *col_idx_ptr_a, const I *row_ptr_b,
    const typename CompressMatrix<I>::IdType *ids_ptr_b, const typename CompressMatrix<I>::BitSetEleType *col_idx_ptr_b,
    const int *use_count, const I *row_ptr_c, int *begins, int *nexts, typename CompressMatrix<I>::IdType *ids,
    typename CompressMatrix<I>::BitSetEleType *values, int *map_use_count) {
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
  // hashmap in LDS
  constexpr int LDS_BEGINS_SIZE = LDS_MAP_CAP;
  constexpr int BLOCK_PROCESS_ONCE = THREADS;

  constexpr int NEXTS_SIZE_PER_BLOCK = BLOCK_PROCESS_ONCE * N;
  constexpr int LDS_NEXTS_SIZE = NEXTS_SIZE_PER_BLOCK;

  __shared__ int shared_begins[LDS_BEGINS_SIZE];
  __shared__ int shared_nexts[LDS_NEXTS_SIZE];
  __shared__ idtype_t shared_ids[LDS_NEXTS_SIZE];
  __shared__ bittype_t shared_values[LDS_NEXTS_SIZE];

  int *block_lds_begins = shared_begins;
  int *block_lds_nexts = shared_nexts;
  idtype_t *block_lds_ids = shared_ids;
  bittype_t *block_lds_values = shared_values;
  // init lds hashmap
  for (int i = block_thread_id; i < LDS_BEGINS_SIZE; i += block_thread_num) {
    shared_begins[i] = HashMap<T, I>::NullValue;
  }
  for (int i = block_thread_id; i < LDS_NEXTS_SIZE; i += block_thread_num) {
    shared_nexts[i] = HashMap<T, I>::NullValue;
  }
  __syncthreads();
  // each vec process one row
  __shared__ int block_lds_use_size;
  __shared__ int block_mem_use_size;

  for (int row_a = global_block_id; row_a < m; row_a += global_block_num) {
    const I row_idx_begin = row_ptr_a[row_a];
    const I row_idx_end = row_ptr_a[row_a + 1];

    int *block_mem_begins = begins + HashMap<bittype_t, idtype_t>::Capacity * row_a;
    int *block_mem_nexts = nexts + row_ptr_c[row_a];
    idtype_t *block_mem_ids = ids + row_ptr_c[row_a];
    bittype_t *block_mem_values = values + row_ptr_c[row_a];

    if (block_thread_id == 0) {
      block_lds_use_size = 0;
      block_mem_use_size = 0;
    }
    __syncthreads();
    int idx_a = row_idx_begin;
    for (; idx_a < row_idx_end; ++idx_a) {
      const I col_a = col_idx_ptr_a[idx_a];
      const I row_b = col_a;
      const I idx_begin_b = row_ptr_b[row_b];
      const idtype_t *block_mem_id_b = ids_ptr_b + idx_begin_b;
      const bittype_t *block_mem_col_b = col_idx_ptr_b + idx_begin_b;
      const int b_count = use_count[row_b];
      const int block_rounds = (b_count) / BLOCK_PROCESS_ONCE + (b_count % BLOCK_PROCESS_ONCE == 0 ? 0 : 1);
      for (int round = 0; round < block_rounds; ++round) {
        if (block_lds_use_size + BLOCK_PROCESS_ONCE > NEXTS_SIZE_PER_BLOCK) {
          // if lds hashmap is full, write to device memory hashmap
          const int end_offset = block_lds_use_size;
          __syncthreads();
          for (int i = block_thread_id; i < end_offset; i += block_thread_num) {
            linear_chain_hashmap_vector_insert_or<bittype_t, idtype_t, HashMap<bittype_t, idtype_t>::HashMask>(
                block_mem_begins, block_mem_nexts, block_mem_ids, block_mem_values, block_lds_ids[i],
                block_lds_values[i], &block_mem_use_size);
          }
          __syncthreads();
          // reset lds hashmap
          for (int i = block_thread_id; i < LDS_MAP_CAP; i += block_thread_num) {
            block_lds_begins[i] = HashMap<T, I>::NullValue;
          }
          for (int i = block_thread_id; i < end_offset; i += block_thread_num) {
            block_lds_nexts[i] = HashMap<T, I>::NullValue;
          }
          if (block_thread_id == 0) {
            block_lds_use_size = 0;
          }
          __syncthreads();
        }
        __syncthreads();
        const int idx_b = BLOCK_PROCESS_ONCE * round + block_thread_id;
        if (idx_b < b_count) {
          // insert to lds hashmap
          linear_chain_hashmap_vector_insert_or<bittype_t, idtype_t, LDS_HASH_MASK>(
              block_lds_begins, block_lds_nexts, block_lds_ids, block_lds_values, block_mem_id_b[idx_b],
              block_mem_col_b[idx_b], &block_lds_use_size);
        }
        __syncthreads();
      }
    }
    __syncthreads();
    // write to device memory hashmap
    const int end_offset = block_lds_use_size;
    __syncthreads();
    for (int i = block_thread_id; i < end_offset; i += block_thread_num) {
      linear_chain_hashmap_vector_insert_or<bittype_t, idtype_t, HashMap<bittype_t, idtype_t>::HashMask>(
          block_mem_begins, block_mem_nexts, block_mem_ids, block_mem_values, block_lds_ids[i], block_lds_values[i],
          &block_mem_use_size);
    }
    __syncthreads();
    // reset lds hashmap
    for (int i = block_thread_id; i < LDS_MAP_CAP; i += block_thread_num) {
      block_lds_begins[i] = HashMap<T, I>::NullValue;
    }
    for (int i = block_thread_id; i < end_offset; i += block_thread_num) {
      block_lds_nexts[i] = HashMap<T, I>::NullValue;
    }
    if (block_thread_id == 0) {
      block_lds_use_size = 0;
    }
    if (block_thread_id == 0) {
      map_use_count[row_a] = block_mem_use_size;
    }
    __syncthreads();
  }
}

template <typename T, typename I, size_t N, size_t LDS_MAP_CAP, size_t LDS_HASH_MASK, size_t THREADS, size_t VEC_SIZE>
__global__ void __launch_bounds__(THREADS) spgemm_symbolic_linear_chain_hashmap_vector_per_row_kernel(
    const size_t m, const I *row_ptr_a, const I *col_idx_ptr_a, const I *row_ptr_b,
    const typename CompressMatrix<I>::IdType *ids_ptr_b, const typename CompressMatrix<I>::BitSetEleType *col_idx_ptr_b,
    const int *use_count, const I *row_ptr_c, int *begins, int *nexts, typename CompressMatrix<I>::IdType *ids,
    typename CompressMatrix<I>::BitSetEleType *values, int *map_use_count) {
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
  // hashmap in LDS
  constexpr int LDS_BEGINS_SIZE = LDS_MAP_CAP * block_vec_num;
  constexpr int VEC_PROCESS_ONCE = VEC_SIZE;

  constexpr int NEXTS_SIZE_PER_VEC = VEC_PROCESS_ONCE * N;
  constexpr int LDS_NEXTS_SIZE = NEXTS_SIZE_PER_VEC * block_vec_num;
  __shared__ int shared_begins[LDS_BEGINS_SIZE];
  __shared__ int shared_nexts[LDS_NEXTS_SIZE];
  __shared__ idtype_t shared_ids[LDS_NEXTS_SIZE];
  __shared__ bittype_t shared_values[LDS_NEXTS_SIZE];
  int *vec_lds_begins = shared_begins + LDS_MAP_CAP * block_vec_id;
  int *vec_lds_nexts = shared_nexts + NEXTS_SIZE_PER_VEC * block_vec_id;
  idtype_t *vec_lds_ids = shared_ids + NEXTS_SIZE_PER_VEC * block_vec_id;
  bittype_t *vec_lds_values = shared_values + NEXTS_SIZE_PER_VEC * block_vec_id;
  // init lds hashmap
  for (int i = block_thread_id; i < LDS_BEGINS_SIZE; i += block_thread_num) {
    shared_begins[i] = HashMap<T, I>::NullValue;
  }
  for (int i = block_thread_id; i < LDS_NEXTS_SIZE; i += block_thread_num) {
    shared_nexts[i] = HashMap<T, I>::NullValue;
  }
  __syncthreads();
  // each vec process one row
  __shared__ int lds_use_size[block_vec_num];
  __shared__ int mem_use_size[block_vec_num];
  int *vec_lds_use_size = lds_use_size + block_vec_id;
  int *vec_mem_use_size = mem_use_size + block_vec_id;

  for (int row_a = global_vec_id; row_a < m; row_a += global_vec_num) {
    const I row_idx_begin = row_ptr_a[row_a];
    const I row_idx_end = row_ptr_a[row_a + 1];

    int *vec_mem_begins = begins + HashMap<bittype_t, idtype_t>::Capacity * row_a;
    int *vec_mem_nexts = nexts + row_ptr_c[row_a];
    idtype_t *vec_mem_ids = ids + row_ptr_c[row_a];
    bittype_t *vec_mem_values = values + row_ptr_c[row_a];

    if (vec_thread_id == 0) {
      *vec_lds_use_size = 0;
      *vec_mem_use_size = 0;
    }
    int idx_a = row_idx_begin;
    for (; idx_a < row_idx_end; ++idx_a) {
      const I col_a = col_idx_ptr_a[idx_a];
      const I row_b = col_a;
      const I idx_begin_b = row_ptr_b[row_b];
      const idtype_t *vec_mem_id_b = ids_ptr_b + idx_begin_b;
      const bittype_t *vec_mem_col_b = col_idx_ptr_b + idx_begin_b;
      const int b_count = use_count[row_b];
      const int vec_rounds = (b_count) / VEC_PROCESS_ONCE + (b_count % VEC_PROCESS_ONCE == 0 ? 0 : 1);
      for (int round = 0; round < vec_rounds; ++round) {
        if (*vec_lds_use_size + VEC_PROCESS_ONCE > NEXTS_SIZE_PER_VEC) {
          // if lds hashmap is full, write to device memory hashmap
          const int end_offset = *vec_lds_use_size;
          for (int i = vec_thread_id; i < end_offset; i += vec_thread_num) {
            linear_chain_hashmap_vector_insert_or<bittype_t, idtype_t, HashMap<bittype_t, idtype_t>::HashMask>(
                vec_mem_begins, vec_mem_nexts, vec_mem_ids, vec_mem_values, vec_lds_ids[i], vec_lds_values[i],
                vec_mem_use_size);
          }
          // reset lds hashmap
          for (int i = vec_thread_id; i < LDS_MAP_CAP; i += vec_thread_num) {
            vec_lds_begins[i] = HashMap<T, I>::NullValue;
          }
          for (int i = vec_thread_id; i < end_offset; i += vec_thread_num) {
            vec_lds_nexts[i] = HashMap<T, I>::NullValue;
          }
          if (vec_thread_id == 0) {
            *vec_lds_use_size = 0;
          }
        }
        const int idx_b = VEC_PROCESS_ONCE * round + vec_thread_id;
        if (idx_b < b_count) {
          // insert to lds hashmap
          linear_chain_hashmap_vector_insert_or<bittype_t, idtype_t, LDS_HASH_MASK>(
              vec_lds_begins, vec_lds_nexts, vec_lds_ids, vec_lds_values, vec_mem_id_b[idx_b], vec_mem_col_b[idx_b],
              vec_lds_use_size);
        }
      }
    }
    // write to device memory hashmap
    const int end_offset = *vec_lds_use_size;
    for (int i = vec_thread_id; i < end_offset; i += vec_thread_num) {
      linear_chain_hashmap_vector_insert_or<bittype_t, idtype_t, HashMap<bittype_t, idtype_t>::HashMask>(
          vec_mem_begins, vec_mem_nexts, vec_mem_ids, vec_mem_values, vec_lds_ids[i], vec_lds_values[i],
          vec_mem_use_size);
    }
    // reset lds hashmap
    for (int i = vec_thread_id; i < LDS_MAP_CAP; i += vec_thread_num) {
      vec_lds_begins[i] = HashMap<T, I>::NullValue;
    }
    for (int i = vec_thread_id; i < end_offset; i += vec_thread_num) {
      vec_lds_nexts[i] = HashMap<T, I>::NullValue;
    }
    if (vec_thread_id == 0) {
      *vec_lds_use_size = 0;
    }
    if (vec_thread_id == 0) {
      map_use_count[row_a] = *vec_mem_use_size;
    }
  }
}

template <typename T, typename I, size_t N, size_t THREADS, size_t VEC_SIZE>
__global__ void __launch_bounds__(THREADS) spgemm_symbolic_linear_chain_hashmap_vector_per_row_one_pass_kernel(
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

  int row_a = global_vec_id;

  if (row_a >= m) {
    return;
  }
  // hashmap in LDS
  constexpr size_t LDS_MAP_CAP = VEC_SIZE * N;
  // hash mask
  constexpr size_t LDS_HASH_MASK = LDS_MAP_CAP - 1;
  constexpr int LDS_BEGINS_SIZE = LDS_MAP_CAP * block_vec_num;
  constexpr int VEC_PROCESS_ONCE = VEC_SIZE;

  constexpr int NEXTS_SIZE_PER_VEC = VEC_PROCESS_ONCE * N;
  constexpr int LDS_NEXTS_SIZE = NEXTS_SIZE_PER_VEC * block_vec_num;
  __shared__ int shared_begins[LDS_BEGINS_SIZE];
  __shared__ int shared_nexts[LDS_NEXTS_SIZE];
  __shared__ idtype_t shared_ids[LDS_NEXTS_SIZE];
  __shared__ bittype_t shared_values[LDS_NEXTS_SIZE];

  int *vec_lds_begins = shared_begins + LDS_MAP_CAP * block_vec_id;
  int *vec_lds_nexts = shared_nexts + NEXTS_SIZE_PER_VEC * block_vec_id;
  idtype_t *vec_lds_ids = shared_ids + NEXTS_SIZE_PER_VEC * block_vec_id;
  bittype_t *vec_lds_values = shared_values + NEXTS_SIZE_PER_VEC * block_vec_id;

  // init lds hashmap
  for (int i = vec_thread_id; i < LDS_MAP_CAP; i += vec_thread_num) {
    vec_lds_begins[i] = HashMap<bittype_t, idtype_t>::NullValue;
  }
  for (int i = vec_thread_id; i < NEXTS_SIZE_PER_VEC; i += vec_thread_num) {
    vec_lds_nexts[i] = HashMap<bittype_t, idtype_t>::NullValue;
  }
  // each vec process one row
  __shared__ int lds_use_size[block_vec_num];
  int *vec_lds_use_size = lds_use_size + block_vec_id;

  const I row_idx_begin = row_ptr_a[row_a];
  const I row_idx_end = row_ptr_a[row_a + 1];

  if (vec_thread_id == 0) {
    *vec_lds_use_size = 0;
  }
  for (int idx_a = row_idx_begin; idx_a < row_idx_end; ++idx_a) {
    const I col_a = col_idx_ptr_a[idx_a];
    const I row_b = col_a;
    const I idx_begin_b = row_ptr_b[row_b];
    const idtype_t *vec_mem_id_b = ids_ptr_b + idx_begin_b;
    const bittype_t *vec_mem_col_b = col_idx_ptr_b + idx_begin_b;
    const int b_count = use_count[row_b];
    for (int i = vec_thread_id; i < b_count; i += vec_thread_num) {
      // insert to lds hashmap
      linear_chain_hashmap_vector_insert_or<bittype_t, idtype_t, LDS_HASH_MASK>(
          vec_lds_begins, vec_lds_nexts, vec_lds_ids, vec_lds_values, vec_mem_id_b[i], vec_mem_col_b[i],
          vec_lds_use_size);
    }
  }
  const int end_offset = *vec_lds_use_size;
  I local_use_count = static_cast<I>(0);
  for (int i = vec_thread_id; i < end_offset; i += vec_thread_num) {
    local_use_count += __popcll(vec_lds_values[i]);
  }
  // reduce sum
  for (int i = VEC_SIZE >> 1; i > 0; i >>= 1) {
    local_use_count += __shfl_down(local_use_count, i, VEC_SIZE);
  }
  if (vec_thread_id == 0) {
    row_ptr_c[row_a + 1] = local_use_count;
  }
}

template <typename T, typename I, size_t N, size_t THREADS>
__global__ void __launch_bounds__(THREADS) spgemm_symbolic_linear_chain_hashmap_block_per_row_one_pass_kernel(
    const size_t m, const I *row_ptr_a, const I *col_idx_ptr_a, const I *row_ptr_b,
    const typename CompressMatrix<I>::IdType *ids_ptr_b, const typename CompressMatrix<I>::BitSetEleType *col_idx_ptr_b,
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
  // hashmap in LDS
  constexpr size_t LDS_MAP_CAP = THREADS * N;
  // hash mask
  constexpr size_t LDS_HASH_MASK = LDS_MAP_CAP - 1;
  constexpr int LDS_BEGINS_SIZE = LDS_MAP_CAP;
  constexpr int BLOCK_PROCESS_ONCE = THREADS;

  constexpr int NEXTS_SIZE_PER_BLOCK = BLOCK_PROCESS_ONCE * N;
  constexpr int LDS_NEXTS_SIZE = NEXTS_SIZE_PER_BLOCK;
  // each block process one row
  int row_a = global_block_id;
  if (row_a >= m) {
    return;
  }

  __shared__ int shared_begins[LDS_BEGINS_SIZE];
  __shared__ int shared_nexts[LDS_NEXTS_SIZE];
  __shared__ idtype_t shared_ids[LDS_NEXTS_SIZE];
  __shared__ bittype_t shared_values[LDS_NEXTS_SIZE];
  __shared__ int block_lds_use_size;
  __shared__ I shared_sum[THREADS / __WF_SIZE__];

  int *block_lds_begins = shared_begins;
  int *block_lds_nexts = shared_nexts;
  idtype_t *block_lds_ids = shared_ids;
  bittype_t *block_lds_values = shared_values;
  // init lds hashmap
  for (int i = block_thread_id; i < LDS_BEGINS_SIZE; i += block_thread_num) {
    shared_begins[i] = HashMap<T, I>::NullValue;
  }
  for (int i = block_thread_id; i < LDS_NEXTS_SIZE; i += block_thread_num) {
    shared_nexts[i] = HashMap<T, I>::NullValue;
  }
  for (int i = block_thread_id; i < THREADS / __WF_SIZE__; i += block_thread_num) {
    shared_sum[i] = static_cast<I>(0);
  }
  if (block_thread_id == 0) {
    block_lds_use_size = 0;
  }
  __syncthreads();

  const I row_idx_begin = row_ptr_a[row_a];
  const I row_idx_end = row_ptr_a[row_a + 1];

  for (int idx_a = row_idx_begin; idx_a < row_idx_end; ++idx_a) {
    const I col_a = col_idx_ptr_a[idx_a];
    const I row_b = col_a;
    const I idx_begin_b = row_ptr_b[row_b];
    const idtype_t *block_mem_id_b = ids_ptr_b + idx_begin_b;
    const bittype_t *block_mem_col_b = col_idx_ptr_b + idx_begin_b;
    const int b_count = use_count[row_b];
    const int block_rounds = b_count / BLOCK_PROCESS_ONCE + (b_count % BLOCK_PROCESS_ONCE == 0 ? 0 : 1);
    for (int round = 0; round < block_rounds; ++round) {
      const int idx_offset = BLOCK_PROCESS_ONCE * round + block_thread_id;
      if (idx_offset < b_count) {
        // insert to lds hashmap
        linear_chain_hashmap_vector_insert_or<bittype_t, idtype_t, LDS_HASH_MASK>(
            block_lds_begins, block_lds_nexts, block_lds_ids, block_lds_values, block_mem_id_b[idx_offset],
            block_mem_col_b[idx_offset], &block_lds_use_size);
      }
    }
    __syncthreads();
  }
  __syncthreads();
  // sort and write to device memory
  const int end_offset = block_lds_use_size;
  I local_use_count = static_cast<I>(0);
  for (int i = block_thread_id; i < end_offset; i += block_thread_num) {
    local_use_count += __popcll(block_lds_values[i]);
  }
  __syncthreads();
  // block sum
  for (int i = __WF_SIZE__ >> 1; i > 0; i >>= 1) {
    local_use_count += __shfl_down(local_use_count, i, __WF_SIZE__);
  }
  if (wf_thread_id == 0) {
    shared_sum[block_wf_id] += local_use_count;
  }
  __syncthreads();
  if (block_wf_id == 0) {
    local_use_count = static_cast<I>(0);
    for (int i = wf_thread_id; i < THREADS / __WF_SIZE__; i += wf_thread_num) {
      local_use_count += shared_sum[i];
    }
    for (int i = __WF_SIZE__ >> 1; i > 0; i >>= 1) {
      local_use_count += __shfl_down(local_use_count, i, __WF_SIZE__);
    }
    if (wf_thread_id == 0) {
      row_ptr_c[row_a + 1] = local_use_count;
    }
  }
}
#endif // SPGEMM_KOKKOS_SPGEMM_SYMBOLIC_KERNEL_HPP
