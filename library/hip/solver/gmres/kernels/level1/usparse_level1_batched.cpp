//
// Created by renyi to 2025-09-16.
//

#include <iostream>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <hip/hip_cooperative_groups.h>

#include "kernels/gmres_optimization_config.h"
#include "include/usparse_config.hpp"
#include "primitives/reduce/device/usparse_reduce.hpp"
#include "usparse_level1_batched.h"
#include "level1/usparse_blas_level1.h"

constexpr int RESTART_TIMES = 20;

namespace usparse {

  template <typename INDEX_TYPE, typename DATA_TYPE, int THREADS>
  __global__ void gs_dot_product_kernel(const DATA_TYPE *d_r0, const DATA_TYPE *d_v, DATA_TYPE *d_h,
                                        DATA_TYPE *d_temp_buffer, const INDEX_TYPE N, const INDEX_TYPE i) {
    const int k = blockIdx.y;
    if (k > i)
      return;

    const DATA_TYPE *d_vk = d_v + k * N;
    const INDEX_TYPE h_index = k * RESTART_TIMES + i;

    const int global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    constexpr int wf_size = usparse::USPARSE_WAVEFRONT_SIZE;
    const int wf_thread_id = threadIdx.x % wf_size;
    const int idx_per_block = (N / gridDim.x) + (N % gridDim.x == 0 ? 0 : 1);
    const int block_idx_begin = blockIdx.x * idx_per_block;
    const int block_idx_end = min(N, blockIdx.x * idx_per_block + idx_per_block);

    DATA_TYPE local_sum = static_cast<DATA_TYPE>(0);

    for (INDEX_TYPE idx = block_idx_begin + threadIdx.x; idx < block_idx_end; idx += blockDim.x) {
      local_sum += d_r0[idx] * d_vk[idx];
    }

    const int tid_in_block = threadIdx.x;
    const DATA_TYPE block_sum =
        usparse::primitives::reduce::block::block_reduce<int, DATA_TYPE, THREADS, usparse::USPARSE_WAVEFRONT_SIZE>(
            local_sum, tid_in_block);

    if (tid_in_block == 0) {
      const int temp_buffer_index = k * gridDim.x + blockIdx.x;
      d_temp_buffer[temp_buffer_index] = block_sum;
    }
  }

  template <typename INDEX_TYPE, typename DATA_TYPE>
  __global__ void gs_daxpy_kernel(DATA_TYPE *d_r0, const DATA_TYPE *d_v, const DATA_TYPE *d_h, const INDEX_TYPE N,
                                  const INDEX_TYPE i) {
    const INDEX_TYPE tid = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (tid >= N)
      return;
    DATA_TYPE r0_val = d_r0[tid];
    for (int k = 0; k <= i; k++) {
      const INDEX_TYPE h_index = k * RESTART_TIMES + i;
      const DATA_TYPE alpha = -d_h[h_index];
      const DATA_TYPE *d_vk = d_v + k * N;
      r0_val += alpha * d_vk[tid];
    }
    d_r0[tid] = r0_val;
  }

  template <typename INDEX_TYPE, typename DATA_TYPE, int THREADS>
  void device_recur(DATA_TYPE *d_temp_buffer, INDEX_TYPE N, DATA_TYPE *d_result);
} // namespace usparse

template <typename INDEX_TYPE, typename DATA_TYPE>
void usparse::gs_dotproduct(const DATA_TYPE *d_r0, const DATA_TYPE *d_v, DATA_TYPE *d_h, DATA_TYPE *d_temp_buffer,
                            const INDEX_TYPE N, const INDEX_TYPE i, std::size_t *temp_buffer_size_byte) {
  constexpr int THREADS = 256;

  const INDEX_TYPE blocks_per_k = N / THREADS + (N % THREADS == 0 ? 0 : 1);
  const INDEX_TYPE total_blocks = (i + 1) * blocks_per_k;

  if (d_temp_buffer == nullptr) {
    *temp_buffer_size_byte = total_blocks * sizeof(DATA_TYPE);
    return;
  } else {
    dim3 grid_dim(blocks_per_k, i + 1);
    (gs_dot_product_kernel<INDEX_TYPE, DATA_TYPE, THREADS>)<<<grid_dim, dim3(THREADS), 0, 0>>>(d_r0, d_v, d_h,
                                                                                               d_temp_buffer, N, i);

    hipDeviceSynchronize();

    for (int k = 0; k <= i; k++) {
      const INDEX_TYPE h_index = k * RESTART_TIMES + i;
      device_recur<INDEX_TYPE, DATA_TYPE, THREADS>(d_temp_buffer + k * blocks_per_k, blocks_per_k, &d_h[h_index]);
    }
  }
}

template <typename INDEX_TYPE, typename DATA_TYPE>
void usparse::gs_daxpy(DATA_TYPE *d_r0, const DATA_TYPE *d_v, const DATA_TYPE *d_h, const INDEX_TYPE N,
                       const INDEX_TYPE i) {
  constexpr int THREADS = 512;
  const int BLOCKS = N / THREADS + (N % THREADS == 0 ? 0 : 1);
  (gs_daxpy_kernel<INDEX_TYPE, DATA_TYPE>)<<<BLOCKS, THREADS, 0, 0>>>(d_r0, d_v, d_h, N, i);
}

template void usparse::gs_dotproduct<uint, double>(const double *d_r0, const double *d_v, double *d_h,
                                                   double *d_temp_buffer, const uint N, const uint i,
                                                   std::size_t *temp_buffer_size_byte);
template void usparse::gs_dotproduct<uint, float>(const float *d_r0, const float *d_v, float *d_h, float *d_temp_buffer,
                                                  const uint N, const uint i, std::size_t *temp_buffer_size_byte);
template void usparse::gs_daxpy<uint, double>(double *d_r0, const double *d_v, const double *d_h, const uint N,
                                              const uint i);
template void usparse::gs_daxpy<uint, float>(float *d_r0, const float *d_v, const float *d_h, const uint N,
                                             const uint i);
