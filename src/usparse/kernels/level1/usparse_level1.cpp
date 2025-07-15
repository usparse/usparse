//
// Created by genshen to 2025-07-09.
//

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "usparse_level1.h"

namespace usparse {

  // todo: pass THREADS as tparam
  template <typename INDEX_TYPE, typename DATA_TYPE>
  __global__ void calculateNorm_kernel(const DATA_TYPE *vec, INDEX_TYPE N, DATA_TYPE *d_temp_buffer) {
    const int global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int wf_size = 64;
    const int wf_thread_id = threadIdx.x % wf_size;
    const int idx_per_block = (N / gridDim.x) + (N % gridDim.x == 0 ? 0 : 1);
    const int block_idx_begin = blockIdx.x * idx_per_block;
    const int block_idx_end = min(N, block_idx_begin + idx_per_block);

    __shared__ DATA_TYPE shared_block_sum;
    if (threadIdx.x == 0) {
      shared_block_sum = static_cast<DATA_TYPE>(0);
    }
    DATA_TYPE local_sum = static_cast<DATA_TYPE>(0);

    for (INDEX_TYPE i = block_idx_begin + threadIdx.x; i < block_idx_end; i += blockDim.x) {
      local_sum += vec[i] * vec[i];
    }
    __syncthreads();
    for (INDEX_TYPE i = wf_size >> 1; i > 0; i >>= 1) {
      local_sum += __shfl_down(local_sum, i, wf_size);
    }
    __syncthreads();
    if (wf_thread_id == 0) {
      atomicAdd(&shared_block_sum, local_sum);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      d_temp_buffer[blockIdx.x] = shared_block_sum;
    }
  }

  template <typename INDEX_TYPE, typename DATA_TYPE>
  __global__ void dotProduct_kernel(const DATA_TYPE *x, const DATA_TYPE *y, const INDEX_TYPE N,
                                    DATA_TYPE *d_temp_buffer) {
    const int global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    constexpr int wf_size = 64;
    const int wf_thread_id = threadIdx.x % wf_size;
    const int idx_per_block = (N / gridDim.x) + (N % gridDim.x == 0 ? 0 : 1);
    const int block_idx_begin = blockIdx.x * idx_per_block;
    const int block_idx_end = min(N, block_idx_begin + idx_per_block);

    __shared__ DATA_TYPE shared_block_sum;
    if (threadIdx.x == 0) {
      shared_block_sum = static_cast<DATA_TYPE>(0);
    }
    DATA_TYPE local_sum = static_cast<DATA_TYPE>(0);
    for (INDEX_TYPE i = block_idx_begin + threadIdx.x; i < block_idx_end; i += blockDim.x) {
      local_sum += x[i] * y[i];
    }
    __syncthreads();
    for (INDEX_TYPE i = wf_size >> 1; i > 0; i >>= 1) {
      local_sum += __shfl_down(local_sum, i, wf_size);
    }
    __syncthreads();
    if (wf_thread_id == 0) {
      atomicAdd(&shared_block_sum, local_sum);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      d_temp_buffer[blockIdx.x] = shared_block_sum;
    }
  }
  template <typename INDEX_TYPE, typename DATA_TYPE>
  __global__ void reduce_kernel(const DATA_TYPE *vec, const INDEX_TYPE N, DATA_TYPE *d_temp_buffer) {
    const int global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int wf_size = 64;
    const int wf_thread_id = threadIdx.x % wf_size;
    const int idx_per_block = (N / gridDim.x) + (N % gridDim.x == 0 ? 0 : 1);
    const int block_idx_begin = blockIdx.x * idx_per_block;
    const int block_idx_end = min(N, block_idx_begin + idx_per_block);

    __shared__ DATA_TYPE shared_block_sum;
    if (threadIdx.x == 0) {
      shared_block_sum = static_cast<DATA_TYPE>(0);
    }
    DATA_TYPE local_sum = static_cast<DATA_TYPE>(0);

    for (INDEX_TYPE i = block_idx_begin + threadIdx.x; i < block_idx_end; i += blockDim.x) {
      local_sum += vec[i];
    }
    __syncthreads();
    for (INDEX_TYPE i = wf_size >> 1; i > 0; i >>= 1) {
      local_sum += __shfl_down(local_sum, i, wf_size);
    }
    __syncthreads();
    if (wf_thread_id == 0) {
      atomicAdd(&shared_block_sum, local_sum);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      d_temp_buffer[blockIdx.x] = shared_block_sum;
    }
  }
  template <typename DATA_TYPE> __global__ void sqrt_kernel(DATA_TYPE *result) { *result = sqrt(*result); }

  template <typename INDEX_TYPE, typename DATA_TYPE>
  __global__ void __launch_bounds__(512)
      daxpy_kernel(const DATA_TYPE alpha, const DATA_TYPE *x, DATA_TYPE *y, const INDEX_TYPE N) {
    const int tid = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (tid < N) {
      y[tid] += alpha * x[tid];
    }
  }

  template <typename INDEX_TYPE, typename DATA_TYPE>
  __global__ void __launch_bounds__(512) dscal_kernel(const DATA_TYPE alpha, DATA_TYPE *x, const INDEX_TYPE N) {
    const int tid = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (tid < N) {
      x[tid] *= alpha;
    }
  }

  template <typename INDEX_TYPE, typename DATA_TYPE>
  __global__ void __launch_bounds__(512) dcopy_kernel(const DATA_TYPE *src, DATA_TYPE *dst, const INDEX_TYPE N) {
    const int tid = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (tid < N) {
      dst[tid] = src[tid];
    }
  }
  template <typename INDEX_TYPE, typename DATA_TYPE>
  void device_recur(DATA_TYPE *d_temp_buffer, INDEX_TYPE N, const int THREADS, DATA_TYPE *d_result) {
    if (N == 1) {
      *d_result = d_temp_buffer[0];
    } else {
      INDEX_TYPE BLOCKS = N / THREADS + (N % THREADS == 0 ? 0 : 1);
      reduce_kernel<<<BLOCKS, THREADS>>>(d_temp_buffer, N, d_temp_buffer);
      hipDeviceSynchronize();
      device_recur(d_temp_buffer, BLOCKS, THREADS, d_result);
    }
  }

} // namespace usparse

template <typename INDEX_TYPE, typename DATA_TYPE>
void usparse::device_calculateNorm(const DATA_TYPE *d_vec, INDEX_TYPE N, DATA_TYPE *d_result) {
  constexpr int THREADS = 256;
  INDEX_TYPE BLOCKS = N / THREADS + (N % THREADS == 0 ? 0 : 1);
  DATA_TYPE *d_temp_buffer;
  hipMalloc(&d_temp_buffer, BLOCKS * sizeof(DATA_TYPE));
  calculateNorm_kernel<<<BLOCKS, THREADS>>>(d_vec, N, d_temp_buffer);
  hipDeviceSynchronize();
  usparse::device_recur(d_temp_buffer, BLOCKS, THREADS, d_result);
  hipFree(d_temp_buffer);
  sqrt_kernel<DATA_TYPE><<<1, 1>>>(d_result);
}
template <typename INDEX_TYPE, typename DATA_TYPE>
void usparse::device_dotProduct(const DATA_TYPE *d_x, const DATA_TYPE *d_y, INDEX_TYPE N, DATA_TYPE *d_result) {
  constexpr int THREADS = 256;
  INDEX_TYPE BLOCKS = N / THREADS + (N % THREADS == 0 ? 0 : 1);
  DATA_TYPE *d_temp_buffer;
  hipMalloc(&d_temp_buffer, BLOCKS * sizeof(DATA_TYPE));
  dotProduct_kernel<<<BLOCKS, THREADS>>>(d_x, d_y, N, d_temp_buffer);
  hipDeviceSynchronize();
  usparse::device_recur(d_temp_buffer, BLOCKS, THREADS, d_result);
  hipFree(d_temp_buffer);
}

template <typename INDEX_TYPE, typename DATA_TYPE>
void usparse::device_daxpy(const DATA_TYPE alpha, const DATA_TYPE *x, DATA_TYPE *y, const INDEX_TYPE N) {
  constexpr int THREADS_PER_BLOCK = 512;
  const int BLOCKS = N / THREADS_PER_BLOCK + (N % THREADS_PER_BLOCK == 0 ? 0 : 1);
  daxpy_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(alpha, x, y, N);
}

template <typename INDEX_TYPE, typename DATA_TYPE>
void usparse::device_dscal(const DATA_TYPE alpha, DATA_TYPE *x, const INDEX_TYPE N) {
  constexpr int THREADS_PER_BLOCK = 512;
  const int BLOCKS = N / THREADS_PER_BLOCK + (N % THREADS_PER_BLOCK == 0 ? 0 : 1);
  dscal_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(alpha, x, N);
}

template <typename INDEX_TYPE, typename DATA_TYPE>
void usparse::device_dcopy(const DATA_TYPE *src, DATA_TYPE *dst, const INDEX_TYPE N) {
  constexpr int THREADS_PER_BLOCK = 512;
  const int BLOCKS = N / THREADS_PER_BLOCK + (N % THREADS_PER_BLOCK == 0 ? 0 : 1);
  dcopy_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(src, dst, N);
}

template void usparse::device_calculateNorm<uint, double>(const double *d_vec, uint N, double *d_result);
template void usparse::device_calculateNorm<uint, float>(const float *d_vec, uint N, float *d_result);

template void usparse::device_dotProduct<uint, double>(const double *d_x, const double *d_y, uint N, double *d_result);
template void usparse::device_dotProduct<uint, float>(const float *d_x, const float *d_y, uint N, float *d_result);

template void usparse::device_daxpy<uint, double>(const double alpha, const double *x, double *y, const uint N);
template void usparse::device_daxpy<uint, float>(const float alpha, const float *x, float *y, const uint N);

template void usparse::device_dscal<uint, double>(const double alpha, double *x, const uint N);
template void usparse::device_dscal<uint, float>(const float alpha, float *x, const uint N);

template void usparse::device_dcopy<uint, double>(const double *src, double *dst, const uint N);
template void usparse::device_dcopy<uint, float>(const float *src, float *dst, const uint N);
