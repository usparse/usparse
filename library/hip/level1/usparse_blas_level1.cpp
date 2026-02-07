//
// Created by genshen to 2025-08-28.
//

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "primitives/reduce/device/usparse_reduce.hpp"
#include "include/usparse_config.hpp"
#include "usparse_blas_level1.h"
#include "usparse_types.h"

namespace usparse {

  template <typename INDEX_TYPE, typename FORM_TYPE, typename TO_TYPE>
  __global__ void __launch_bounds__(512) cast_kernel(const FORM_TYPE *src, TO_TYPE *dst, const INDEX_TYPE N) {
    const int tid = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (tid < N) {
      dst[tid] = static_cast<TO_TYPE>(src[tid]);
    }
  }

  template <typename INDEX_TYPE, typename DATA_TYPE>
  __global__ void __launch_bounds__(512) memset_kernel(DATA_TYPE *src, const INDEX_TYPE N) {
    const int tid = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (tid < N) {
      src[tid] = static_cast<DATA_TYPE>(0);
    }
  }

  // todo: pass THREADS as tparam
  template <typename INDEX_TYPE, typename DATA_TYPE, int THREADS>
  __global__ void calculateNorm_kernel(const DATA_TYPE *vec, INDEX_TYPE N, DATA_TYPE *d_temp_buffer) {
    const int global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    constexpr int wf_size = usparse::USPARSE_WAVEFRONT_SIZE;
    const int wf_thread_id = threadIdx.x % wf_size;
    const int idx_per_block = (N / gridDim.x) + (N % gridDim.x == 0 ? 0 : 1);
    const int block_idx_begin = blockIdx.x * idx_per_block;
    const int block_idx_end = min(N, block_idx_begin + idx_per_block);

    DATA_TYPE local_sum = static_cast<DATA_TYPE>(0);

    for (INDEX_TYPE i = block_idx_begin + threadIdx.x; i < block_idx_end; i += blockDim.x) {
      local_sum += vec[i] * vec[i];
    }

    // reduce value in a block into the first thread of the block.
    const int tid_in_block = threadIdx.x;
    const DATA_TYPE block_sum =
        usparse::line_enhance_block_reduce<int, DATA_TYPE, THREADS, usparse::USPARSE_WAVEFRONT_SIZE>(local_sum,
                                                                                                     tid_in_block);

    if (tid_in_block == 0) {
      d_temp_buffer[blockIdx.x] = block_sum;
    }
  }

  template <typename INDEX_TYPE, typename DATA_TYPE, int THREADS>
  __global__ void dotProduct_kernel(const DATA_TYPE *x, const DATA_TYPE *y, const INDEX_TYPE N,
                                    DATA_TYPE *d_temp_buffer) {
    const int global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    constexpr int wf_size = usparse::USPARSE_WAVEFRONT_SIZE;
    const int wf_thread_id = threadIdx.x % wf_size;
    const int idx_per_block = (N / gridDim.x) + (N % gridDim.x == 0 ? 0 : 1);
    const int block_idx_begin = blockIdx.x * idx_per_block;
    const int block_idx_end = min(N, block_idx_begin + idx_per_block);

    DATA_TYPE local_sum = static_cast<DATA_TYPE>(0);
    for (INDEX_TYPE i = block_idx_begin + threadIdx.x; i < block_idx_end; i += blockDim.x) {
      local_sum += x[i] * y[i];
    }

    // reduce value in a block into the first thread of the block.
    const int tid_in_block = threadIdx.x;
    const DATA_TYPE block_sum =
        usparse::line_enhance_block_reduce<int, DATA_TYPE, THREADS, usparse::USPARSE_WAVEFRONT_SIZE>(local_sum,
                                                                                                     tid_in_block);

    if (tid_in_block == 0) {
      d_temp_buffer[blockIdx.x] = block_sum;
    }
  }
  template <typename INDEX_TYPE, typename DATA_TYPE, int THREADS>
  __global__ void reduce_kernel(const DATA_TYPE *vec, const INDEX_TYPE N, DATA_TYPE *d_temp_buffer) {
    const int global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    constexpr int wf_size = usparse::USPARSE_WAVEFRONT_SIZE;
    const int wf_thread_id = threadIdx.x % wf_size;
    const int idx_per_block = (N / gridDim.x) + (N % gridDim.x == 0 ? 0 : 1);
    const int block_idx_begin = blockIdx.x * idx_per_block;
    const int block_idx_end = min(N, block_idx_begin + idx_per_block);

    DATA_TYPE local_sum = static_cast<DATA_TYPE>(0);
    for (INDEX_TYPE i = block_idx_begin + threadIdx.x; i < block_idx_end; i += blockDim.x) {
      local_sum += vec[i];
    }

    // reduce value in a block into the first thread of the block.
    const int tid_in_block = threadIdx.x;
    const DATA_TYPE block_sum =
        usparse::line_enhance_block_reduce<int, DATA_TYPE, THREADS, usparse::USPARSE_WAVEFRONT_SIZE>(local_sum,
                                                                                                     tid_in_block);

    if (tid_in_block == 0) {
      d_temp_buffer[blockIdx.x] = block_sum;
    }
  }

  template <typename DATA_TYPE> __global__ void sqrt_kernel(DATA_TYPE *result) { *result = sqrt(*result); }

  template <typename INDEX_TYPE, typename F_TYPE, typename DATA_TYPE>
  __global__ void __launch_bounds__(512)
      daxpy_kernel(const DATA_TYPE alpha, const F_TYPE *x, DATA_TYPE *y, const INDEX_TYPE N) {
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

  template <typename INDEX_TYPE, typename DATA_TYPE, int THREADS>
  __global__ void __launch_bounds__(512) sacle_set_kernel(const DATA_TYPE alpha, DATA_TYPE data[], const INDEX_TYPE N) {
    const int tid = THREADS * hipBlockIdx_x + hipThreadIdx_x;
    if (tid < N) {
      data[tid] = alpha;
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
  __global__ void __launch_bounds__(512)
      scale_dcopy_kernel(const DATA_TYPE *src, DATA_TYPE *dst, const DATA_TYPE alpha, const INDEX_TYPE N) {
    const int tid = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (tid < N) {
      dst[tid] = alpha * src[tid];
    }
  }

  template <typename INDEX_TYPE, typename DATA_TYPE, int THREADS>
  void device_recur(DATA_TYPE *d_temp_buffer, INDEX_TYPE N, DATA_TYPE *d_result) {
    if (N == 1) {
      *d_result = d_temp_buffer[0];
    } else {
      INDEX_TYPE BLOCKS = N / THREADS + (N % THREADS == 0 ? 0 : 1);
      (reduce_kernel<INDEX_TYPE, DATA_TYPE, THREADS>)<<<BLOCKS, THREADS>>>(d_temp_buffer, N, d_temp_buffer);
      hipDeviceSynchronize();
      (device_recur<INDEX_TYPE, DATA_TYPE, THREADS>)(d_temp_buffer, BLOCKS, d_result);
    }
  }
} // namespace usparse

template <typename INDEX_TYPE, typename DATA_TYPE>
void usparse::device_calculateNorm(const DATA_TYPE *d_vec, INDEX_TYPE N, DATA_TYPE *d_result, DATA_TYPE *d_temp_buffer,
                                   std::size_t *temp_buffer_size_byte) {
  constexpr int THREADS = 256;
  INDEX_TYPE BLOCKS = N / THREADS + (N % THREADS == 0 ? 0 : 1);
  if (d_temp_buffer == nullptr) {
    *temp_buffer_size_byte = BLOCKS * sizeof(DATA_TYPE);
    return;
  } else {
    (calculateNorm_kernel<INDEX_TYPE, DATA_TYPE, THREADS>)<<<BLOCKS, THREADS>>>(d_vec, N, d_temp_buffer);
    usparse::device_recur<INDEX_TYPE, DATA_TYPE, THREADS>(d_temp_buffer, BLOCKS, d_result);
    sqrt_kernel<DATA_TYPE><<<1, 1>>>(d_result);
    return;
  }
}

template <typename INDEX_TYPE, typename DATA_TYPE>
void usparse::device_dotProduct(const DATA_TYPE *d_x, const DATA_TYPE *d_y, INDEX_TYPE N, DATA_TYPE *d_result,
                                DATA_TYPE *d_temp_buffer, std::size_t *temp_buffer_size_byte) {
  constexpr int THREADS = 256;
  INDEX_TYPE BLOCKS = N / THREADS + (N % THREADS == 0 ? 0 : 1);
  if (d_temp_buffer == nullptr) {
    *temp_buffer_size_byte = BLOCKS * sizeof(DATA_TYPE);
    return;
  } else {
    (dotProduct_kernel<INDEX_TYPE, DATA_TYPE, THREADS>)<<<BLOCKS, THREADS>>>(d_x, d_y, N, d_temp_buffer);
    usparse::device_recur<INDEX_TYPE, DATA_TYPE, THREADS>(d_temp_buffer, BLOCKS, d_result);
    return;
  }
}

template <typename INDEX_TYPE, typename F_TYPE, typename DATA_TYPE>
void usparse::device_daxpy(const DATA_TYPE alpha, const F_TYPE *x, DATA_TYPE *y, const INDEX_TYPE N) {
  constexpr int THREADS_PER_BLOCK = 512;
  const int BLOCKS = N / THREADS_PER_BLOCK + (N % THREADS_PER_BLOCK == 0 ? 0 : 1);
  daxpy_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(alpha, x, y, N);
}

template <typename INDEX_TYPE, typename DATA_TYPE>
void usparse::device_dscal(const DATA_TYPE alpha, DATA_TYPE *x, const INDEX_TYPE N, hipStream_t stream) {
  constexpr int THREADS_PER_BLOCK = 512;
  const int BLOCKS = N / THREADS_PER_BLOCK + (N % THREADS_PER_BLOCK == 0 ? 0 : 1);
  dscal_kernel<<<BLOCKS, THREADS_PER_BLOCK, 0, stream>>>(alpha, x, N);
}

template <typename INDEX_TYPE, typename DATA_TYPE>
void usparse::device_dcopy(const DATA_TYPE *src, DATA_TYPE *dst, const INDEX_TYPE N) {
  constexpr int THREADS_PER_BLOCK = 512;
  const int BLOCKS = N / THREADS_PER_BLOCK + (N % THREADS_PER_BLOCK == 0 ? 0 : 1);
  dcopy_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(src, dst, N);
}

template <typename INDEX_TYPE, typename DATA_TYPE>
void usparse::device_copy_v2(const DATA_TYPE *src, DATA_TYPE *dst, const DATA_TYPE alpha, const INDEX_TYPE N) {
  constexpr int THREADS_PER_BLOCK = 512;
  const int BLOCKS = N / THREADS_PER_BLOCK + (N % THREADS_PER_BLOCK == 0 ? 0 : 1);
  scale_dcopy_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(src, dst, alpha, N);
}

template <typename INDEX_TYPE, typename FORM_TYPE, typename TO_TYPE>
void usparse::device_cast(const FORM_TYPE *src, TO_TYPE *dst, const INDEX_TYPE N) {
  constexpr int THREADS_PER_BLOCK = 512;
  const int BLOCKS = N / THREADS_PER_BLOCK + (N % THREADS_PER_BLOCK == 0 ? 0 : 1);
  cast_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(src, dst, N);
}

template <typename INDEX_TYPE, typename DATA_TYPE> void usparse::device_memset(DATA_TYPE *src, const INDEX_TYPE N) {
  constexpr int THREADS_PER_BLOCK = 512;
  const int BLOCKS = N / THREADS_PER_BLOCK + (N % THREADS_PER_BLOCK == 0 ? 0 : 1);
  memset_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(src, N);
}

template <typename INDEX_TYPE, typename DATA_TYPE>
void usparse::device_scale_set(const DATA_TYPE alpha, DATA_TYPE *data, const INDEX_TYPE N, hipStream_t stream) {
  constexpr unsigned int THREADS = 256;
  const unsigned int grid_size = (N + THREADS - 1) / THREADS;

  (sacle_set_kernel<INDEX_TYPE, DATA_TYPE, THREADS>)<<<grid_size, THREADS, 0, stream>>>(alpha, data, N);
}

template void usparse::device_calculateNorm<uint, double>(const double *d_vec, uint N, double *d_result,
                                                          double *d_temp_buffer, std::size_t *temp_buffer_size_byte);
template void usparse::device_calculateNorm<uint, float>(const float *d_vec, uint N, float *d_result,
                                                         float *d_temp_buffer, std::size_t *temp_buffer_size_byte);

template void usparse::device_dotProduct<uint, double>(const double *d_x, const double *d_y, uint N, double *d_result,
                                                       double *d_temp_buffer, std::size_t *temp_buffer_size_byte);
template void usparse::device_dotProduct<uint, float>(const float *d_x, const float *d_y, uint N, float *d_result,
                                                      float *d_temp_buffer, std::size_t *temp_buffer_size_byte);

template void usparse::device_daxpy<uint, float, double>(const double alpha, const float *x, double *y, const uint N);
template void usparse::device_daxpy<uint, float, float>(const float alpha, const float *x, float *y, const uint N);
template void usparse::device_daxpy<uint, double, double>(const double alpha, const double *x, double *y, const uint N);

template void usparse::device_dscal<uint, double>(const double alpha, double *x, const uint N, hipStream_t stream);
template void usparse::device_dscal<uint, float>(const float alpha, float *x, const uint N, hipStream_t stream);
template void usparse::device_dscal<int, int>(const int alpha, int *x, const int N, hipStream_t stream);
template void usparse::device_dscal<int, double>(const double alpha, double *x, const int N, hipStream_t stream);
template void usparse::device_dscal<int, float>(const float alpha, float *x, const int N, hipStream_t stream);
template void usparse::device_dscal<int, usparse::usparse_fpc64>(const usparse::usparse_fpc64 alpha,
                                                                 usparse::usparse_fpc64 *x, const int N,
                                                                 hipStream_t stream);
template void usparse::device_dscal<int, usparse::usparse_fpc32>(const usparse::usparse_fpc32 alpha,
                                                                 usparse::usparse_fpc32 *x, const int N,
                                                                 hipStream_t stream);
template void usparse::device_dscal<int64_t, int32_t>(const int32_t alpha, int32_t *x, const int64_t N,
                                                      hipStream_t stream);
template void usparse::device_dscal<int64_t, usparse::usparse_fp64>(const usparse::usparse_fp64 alpha,
                                                                    usparse::usparse_fp64 *x, const int64_t N,
                                                                    hipStream_t stream);
template void usparse::device_dscal<int64_t, usparse::usparse_fp32>(const usparse::usparse_fp32 alpha,
                                                                    usparse::usparse_fp32 *x, const int64_t N,
                                                                    hipStream_t stream);
template void usparse::device_dscal<int64_t, usparse::usparse_fpc64>(const usparse::usparse_fpc64 alpha,
                                                                     usparse::usparse_fpc64 *x, const int64_t N,
                                                                     hipStream_t stream);
template void usparse::device_dscal<int64_t, usparse::usparse_fpc32>(const usparse::usparse_fpc32 alpha,
                                                                     usparse::usparse_fpc32 *x, const int64_t N,
                                                                     hipStream_t stream);

template void usparse::device_scale_set<uint, double>(const double alpha, double *data, const uint N,
                                                      hipStream_t stream);
template void usparse::device_scale_set<uint, float>(const float alpha, float *data, const uint N, hipStream_t stream);
template void usparse::device_scale_set<int, int>(const int alpha, int *data, const int N, hipStream_t stream);
template void usparse::device_scale_set<int, double>(const double alpha, double *data, const int N, hipStream_t stream);
template void usparse::device_scale_set<int, float>(const float alpha, float *data, const int N, hipStream_t stream);
template void usparse::device_scale_set<int, usparse::usparse_fpc64>(const usparse::usparse_fpc64 alpha,
                                                                     usparse::usparse_fpc64 *data, const int N,
                                                                     hipStream_t stream);
template void usparse::device_scale_set<int, usparse::usparse_fpc32>(const usparse::usparse_fpc32 alpha,
                                                                     usparse::usparse_fpc32 *data, const int N,
                                                                     hipStream_t stream);
template void usparse::device_scale_set<int64_t, int32_t>(const int32_t alpha, int32_t *data, const int64_t N,
                                                          hipStream_t stream);
template void usparse::device_scale_set<int64_t, usparse::usparse_fp64>(const usparse::usparse_fp64 alpha,
                                                                        usparse::usparse_fp64 *data, const int64_t N,
                                                                        hipStream_t stream);
template void usparse::device_scale_set<int64_t, usparse::usparse_fp32>(const usparse::usparse_fp32 alpha,
                                                                        usparse::usparse_fp32 *data, const int64_t N,
                                                                        hipStream_t stream);
template void usparse::device_scale_set<int64_t, usparse::usparse_fpc64>(const usparse::usparse_fpc64 alpha,
                                                                         usparse::usparse_fpc64 *data, const int64_t N,
                                                                         hipStream_t stream);
template void usparse::device_scale_set<int64_t, usparse::usparse_fpc32>(const usparse::usparse_fpc32 alpha,
                                                                         usparse::usparse_fpc32 *data, const int64_t N,
                                                                         hipStream_t stream);

template void usparse::device_dcopy<uint, double>(const double *src, double *dst, const uint N);
template void usparse::device_dcopy<uint, float>(const float *src, float *dst, const uint N);

template void usparse::device_copy_v2<uint, double>(const double *src, double *dst, const double alpha, const uint N);
template void usparse::device_copy_v2<uint, float>(const float *src, float *dst, const float alpha, const uint N);

template void usparse::device_cast<uint, double, float>(const double *src, float *dst, const uint N);
template void usparse::device_cast<uint, float, double>(const float *src, double *dst, const uint N);

template void usparse::device_memset<uint, double>(double *src, const uint N);
template void usparse::device_memset<uint, float>(float *src, const uint N);
