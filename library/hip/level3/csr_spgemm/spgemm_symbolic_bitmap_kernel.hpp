//
// Created by luxu on 2023/02/14.
//
#ifndef USTB_SPARSE_SPGEMM_SPGEMM_SYMBOLIC_BITMAP_KERNEL_HPP
#define USTB_SPARSE_SPGEMM_SPGEMM_SYMBOLIC_BITMAP_KERNEL_HPP

#include "compress_matrix.h"
#include <hip/hip_runtime.h>

template <typename I, size_t CHUNK_SIZE>
__global__ void create_compress_bp_wf_per_row_kernel(const size_t m, const I *row_ptr,
                                                     typename CompressMatrix<I>::IdType *ids, int *use_count, I *bp,
                                                     const size_t BP_IDX_PER_ROW) {
  // thread id in global
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  // thread num in global
  const int global_thread_num = gridDim.x * blockDim.x;
  // wf id in global
  const int global_wf_id = global_thread_id / __WF_SIZE__;
  // wf num in global
  const int global_wf_num = global_thread_num / __WF_SIZE__;
  // thread id in wf
  const int wf_thread_id = global_thread_id % __WF_SIZE__;
  // thread num in wf
  const int wf_thread_num = __WF_SIZE__;

  // each wf process a row
  for (int row = global_wf_id; row < m; row += global_wf_num) {
    I *row_bp_ptr = bp + BP_IDX_PER_ROW * row;
    I base_bp = CHUNK_SIZE;
    int bp_idx = 1;
    const I row_idx_begin = row_ptr[row];
    const I row_idx_end = row_idx_begin + use_count[row];
    const int WF_ROUNDS = (use_count[row] + __WF_SIZE__ - 1) / __WF_SIZE__;
    for (int i = 0; i < WF_ROUNDS; i++) {
      const int idx = row_idx_begin + __WF_SIZE__ * i + wf_thread_id;
      const I col = idx < row_idx_end ? ids[idx] : static_cast<I>(0);
      bool flag = col >= base_bp;
      if (__any(flag)) {
        do {
          I tmp_idx = flag ? idx : row_idx_end;
          for (int i = __WF_SIZE__ >> 1; i > 0; i >>= 1) {
            tmp_idx = min(__shfl_down(tmp_idx, i), tmp_idx);
          }
          tmp_idx = __shfl(tmp_idx, 0);
          I tmp_col = ids[tmp_idx];
          const int bp_idx_begin = bp_idx;
          const int bp_idx_end = tmp_col / CHUNK_SIZE + 1;
          for (int i = bp_idx_begin + wf_thread_id; i < bp_idx_end; i += wf_thread_num) {
            row_bp_ptr[i] = tmp_idx;
          }
          bp_idx = bp_idx_end;
          base_bp = bp_idx_end * CHUNK_SIZE;
          flag = col >= base_bp;
        } while (__any(flag));
      }
    }
    if (wf_thread_id == 0) {
      row_bp_ptr[0] = row_idx_begin;
    }
    bp_idx = __shfl(bp_idx, 0);
    for (int idx = bp_idx + wf_thread_id; idx < BP_IDX_PER_ROW; idx += wf_thread_num) {
      row_bp_ptr[idx] = row_idx_end;
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

template <typename T, typename I, size_t N, size_t THREADS>
__global__ void __launch_bounds__(THREADS)
    spgemm_symbolic_block_per_row_multi_pass_kernel(const size_t m, const I *row_ptr_a, const I *col_idx_ptr_a,
                                                    const I *row_ptr_b,
                                                    const typename CompressMatrix<I>::IdType *ids_ptr_b,
                                                    const typename CompressMatrix<I>::BitSetEleType *col_idx_ptr_b,
                                                    const int *use_count, I *bp, const size_t BP_IDX_PER_ROW, const size_t BPS_PER_ROW,
                                                    I *row_ptr_c) {
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
  const I row_idx_begin = row_ptr_a[row_a];
  const I row_idx_end = row_ptr_a[row_a + 1];

  for (int r = 0; r < BPS_PER_ROW; r++) {
    // init lds hashmap
    for (int i = block_thread_id; i < BLOCK_LDS_SIZE; i += block_thread_num) {
      shared_values[i] = static_cast<bittype_t>(0);
    }
    if (block_thread_id == 0) {
      shared_nnz_count = static_cast<I>(0);
    }
    __syncthreads();
    for (int idx_a = row_idx_begin + block_wf_id; idx_a < row_idx_end; idx_a += block_wf_num) {
      const I col_a = col_idx_ptr_a[idx_a];
      const I row_b = col_a;
      const I idx_begin_b = bp[row_b * BP_IDX_PER_ROW + r];
      const I idx_end_b = bp[row_b * BP_IDX_PER_ROW + r + 1];
      const idtype_t *block_mem_id_b = ids_ptr_b + idx_begin_b;
      const bittype_t *block_mem_col_b = col_idx_ptr_b + idx_begin_b;
      for (int idx_b = wf_thread_id + idx_begin_b; idx_b < idx_end_b; idx_b += wf_thread_num) {
        atomicOr(block_lds_values + (ids_ptr_b[idx_b] % BLOCK_LDS_SIZE), col_idx_ptr_b[idx_b]);
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
      row_ptr_c[row_a + 1] += shared_nnz_count;
    }
  }
}

#endif // USTB_SPARSE_SPGEMM_SPGEMM_SYMBOLIC_BITMAP_KERNEL_HPP
