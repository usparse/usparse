#ifndef SPGEMM_KOKKOS_SPGEMM_SYMBOLIC_BIT_MAP_KERNEL_HPP
#define SPGEMM_KOKKOS_SPGEMM_SYMBOLIC_BIT_MAP_KERNEL_HPP

#include <hip/hip_runtime.h>

// without hash
template <typename T, typename I, typename bitMapType, size_t THREADS, size_t VEC_SIZE>
__global__ void __launch_bounds__(THREADS)
    spgemm_symbolic_bit_map_wf_per_row(const size_t m, const I *row_ptr_a, const I *col_idx_ptr_a, const I *row_ptr_b,
                                       const I *col_idx_ptr_b, I *row_ptr_c) {
  // thread id in global
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  // wf id in global
  const int global_wf_id = global_thread_id / __WF_SIZE__;
  // wf id in block
  const int block_wf_id = threadIdx.x / __WF_SIZE__;
  // thread id in wf
  const int wf_thread_id = global_thread_id % __WF_SIZE__;
  // vec id in wf
  const int wf_vec_id = wf_thread_id / VEC_SIZE;
  // thread id in vec
  const int vec_thread_id = global_thread_id % VEC_SIZE;
  // thread num in wf
  constexpr int wf_thread_num = __WF_SIZE__;
  // vec num in wf
  const int wf_vec_num = wf_thread_num / VEC_SIZE;
  // thread num in vec
  constexpr int vec_thread_num = VEC_SIZE;

  const int row_a = global_wf_id;
  if (row_a >= m) {
    return;
  }

  extern __shared__ bitMapType shared_bit_map[];
  // __shared__ bitMapType shared_bit_map[];

  constexpr size_t BitsLen = sizeof(bitMapType) * 8;
  const size_t WF_LDS_NUM = 512;
  bitMapType *wf_lds_ids = shared_bit_map + WF_LDS_NUM * block_wf_id;

  // init lds hashmap
  for (int i = wf_thread_id; i < WF_LDS_NUM; i += wf_thread_num) {
    wf_lds_ids[i] = static_cast<bitMapType>(0);
  }
  const I row_idx_begin_a = row_ptr_a[row_a];
  const I row_idx_end_a = row_ptr_a[row_a + 1];

  for (int idx_a = row_idx_begin_a + wf_vec_id; idx_a < row_idx_end_a; idx_a += wf_vec_num) {
    const I col_a = col_idx_ptr_a[idx_a];
    const I row_b = col_a;
    const I row_idx_begin_b = row_ptr_b[row_b];
    const I row_idx_end_b = row_ptr_b[row_b + 1];

    for (int idx_b = row_idx_begin_b + vec_thread_id; idx_b < row_idx_end_b; idx_b += vec_thread_num) {
      I col_b = col_idx_ptr_b[idx_b];
      bitMapType mask = static_cast<bitMapType>(0x1) << (col_b % BitsLen);
      atomicOr(wf_lds_ids + col_b / BitsLen, mask);
      // wf_lds_ids[col_b / BitsLen] |= mask;
    }
  }

  int local_use_count = static_cast<int>(0);
  for (int i = wf_thread_id; i < WF_LDS_NUM - __WF_SIZE__; i += wf_thread_num * 2) {
    local_use_count += __popcll(wf_lds_ids[i]);
    local_use_count += __popcll(wf_lds_ids[i + __WF_SIZE__]);
  }

  for (int i = __WF_SIZE__ >> 1; i > 0; i >>= 1) {
    local_use_count += __shfl_down(local_use_count, i, __WF_SIZE__);
  }

  if (wf_thread_id == 0) {
    row_ptr_c[row_a + 1] = local_use_count;
  }
}

template <typename T, typename I, typename bitMapType, size_t THREADS>
__global__ void __launch_bounds__(THREADS)
    spgemm_symbolic_bit_map_block_per_row(const size_t m, const I *row_ptr_a, const I *col_idx_ptr_a,
                                          const I *row_ptr_b, const I *col_idx_ptr_b, I *row_ptr_c) {
  // block id in global
  const int global_block_id = blockIdx.x;
  // thread id in block
  const int block_thread_id = threadIdx.x;
  // thread num in block
  const int block_thread_num = blockDim.x;
  // wf id in block
  const int block_wf_id = threadIdx.x / __WF_SIZE__;
  // wf num in block
  constexpr int block_wf_num = THREADS / __WF_SIZE__;
  // thread id in wf
  const int wf_thread_id = threadIdx.x % __WF_SIZE__;
  // thread num in wf
  const int wf_thread_num = __WF_SIZE__;

  const int row_a = global_block_id;
  if (row_a >= m) {
    return;
  }

  extern __shared__ bitMapType block_lds_ids[];

  __shared__ int offset;
  offset = 0;

  constexpr size_t BitsLen = sizeof(bitMapType) * 8;
  const size_t BLOCK_LDS_NUM = m / BitsLen + (m % BitsLen == 0 ? 0 : 1);

  // init lds hashmap
  for (int i = block_thread_id; i < BLOCK_LDS_NUM; i += block_thread_num) {
    block_lds_ids[i] = static_cast<bitMapType>(0);
  }
  const I row_idx_begin_a = row_ptr_a[row_a];
  const I row_idx_end_a = row_ptr_a[row_a + 1];

  for (int idx_a = row_idx_begin_a + block_wf_id; idx_a < row_idx_end_a; idx_a += block_wf_num) {
    const I col_a = col_idx_ptr_a[idx_a];
    const I row_b = col_a;
    const I row_idx_begin_b = row_ptr_b[row_b];
    const I row_idx_end_b = row_ptr_b[row_b + 1];

    for (int idx_b = row_idx_begin_b + wf_thread_id; idx_b < row_idx_end_b; idx_b += wf_thread_num) {
      I col_b = col_idx_ptr_b[idx_b];
      bitMapType mask = static_cast<bitMapType>(0x1) << (col_b % BitsLen);
      atomicOr(block_lds_ids + col_b / BitsLen, mask);
    }
  }
  __syncthreads();

  int local_use_count = static_cast<int>(0);
  for (int i = block_thread_id; i < BLOCK_LDS_NUM; i += block_thread_num) {
    local_use_count += __popcll(block_lds_ids[i]);
  }
  __syncthreads();

  for (int i = __WF_SIZE__ >> 1; i > 0; i >>= 1) {
    local_use_count += __shfl_down(local_use_count, i, __WF_SIZE__);
  }
  __syncthreads();

  if (wf_thread_id == 0) {
    atomicAdd(&offset, local_use_count);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    row_ptr_c[row_a + 1] = offset;
  }
}

#endif // SPGEMM_KOKKOS_SPGEMM_SYMBOLIC_KERNEL_HPP
