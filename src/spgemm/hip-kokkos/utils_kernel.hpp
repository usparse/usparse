#ifndef SPGEMM_KOKKOS_UTILS_KERNEL_HPP
#define SPGEMM_KOKKOS_UTILS_KERNEL_HPP

// impl of scan than fan
template <typename T, size_t VEC_SIZE, size_t R, size_t THREADS, size_t WF_SIZE>
__global__ void prefix_sum_kernel(const size_t len, T *array, T *buffer) {
  // thread id in global
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  // thread num in global
  const int global_thread_num = blockDim.x * gridDim.x;
  // block id in global
  const int global_block_id = blockIdx.x;
  // block num in global
  const int global_block_num = gridDim.x;
  // thread id in block
  const int block_thread_id = threadIdx.x;
  // thread num in block
  constexpr int block_thread_num = THREADS;
  // vector id in block
  const int block_vector_id = block_thread_id / VEC_SIZE;
  // vector num in block
  constexpr int block_vector_num = THREADS / VEC_SIZE;
  // thread id in vector
  const int vector_thread_id = block_thread_id % VEC_SIZE;
  // thread num in vector
  constexpr int vector_thread_num = VEC_SIZE;

  __shared__ T shared_block_scan[block_vector_num * R];
  const int process_each_block = THREADS * R;
  const int block_scan_begin = global_block_id * process_each_block;
  const int block_scan_end = min(block_scan_begin + process_each_block, len);
  T vec_scan[R];
#pragma unroll
  for (int i = 0; i < R; ++i) {
    const int vector_scan_begin = block_scan_begin + block_vector_id * VEC_SIZE + THREADS * i;
    const int arr_idx = vector_scan_begin + vector_thread_id;
    T sum = arr_idx < block_scan_end ? array[arr_idx] : 0;
// wavefront reduce
#pragma unroll
    for (int j = 1; j < VEC_SIZE; j <<= 1) {
      T tmp = __shfl_up(sum, j, VEC_SIZE);
      if (vector_thread_id >= j) {
        sum += tmp;
      }
    }
    vec_scan[i] = sum;
    if (vector_thread_id == VEC_SIZE - 1) {
      shared_block_scan[block_vector_num * i + block_vector_id] = sum;
    }
  }
  __syncthreads();
  // block_reduce
  if (block_vector_id == 0) {
    T block_sum = static_cast<T>(0);
    constexpr int rounds = block_vector_num * R / VEC_SIZE + ((block_vector_num * R) % VEC_SIZE == 0 ? 0 : 1);
#pragma unroll
    for (int i = 0; i < rounds; ++i) {
      const int idx = i * VEC_SIZE + vector_thread_id;
      // wavefront reduce
      T sum = idx < block_vector_num * R ? shared_block_scan[idx] : 0;
#pragma unroll
      for (int j = 1; j < VEC_SIZE; j <<= 1) {
        T tmp = __shfl_up(sum, j, VEC_SIZE);
        if (vector_thread_id >= j) {
          sum += tmp;
        }
      }
      if (idx < block_vector_num * R) {
        shared_block_scan[idx] = block_sum + sum;
      }
      if ((i + 1) * VEC_SIZE - 1 < block_vector_num * R) {
        block_sum = shared_block_scan[(i + 1) * VEC_SIZE - 1];
      } else {
        block_sum = shared_block_scan[block_vector_num * R - 1];
      }
    }
    // write back
    if (vector_thread_id == VEC_SIZE - 1) {
      buffer[global_block_id] = block_sum;
    }
  }
  __syncthreads();
  for (int i = 0; i < R; ++i) {
    const int vector_fan_begin = block_scan_begin + block_vector_id * VEC_SIZE + THREADS * i;
    const int arr_idx = vector_fan_begin + vector_thread_id;
    T sum = vec_scan[i];
    if (block_vector_num * i + block_vector_id > 0) {
      sum += shared_block_scan[block_vector_num * i + block_vector_id - 1];
    }
    // wavefront reduce
    if (arr_idx < block_scan_end) {
      array[arr_idx] = sum;
    }
  }
}

template <typename T, size_t VEC_SIZE, size_t THREADS, size_t WF_SIZE>
__global__ void buffer_reduce_kernel(const size_t len, T *buffer) {
  // thread id in global
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  // thread num in global
  const int global_thread_num = blockDim.x * gridDim.x;
  // block id in global
  const int global_block_id = blockIdx.x;
  // block num in global
  const int global_block_num = gridDim.x;
  // thread id in block
  const int block_thread_id = threadIdx.x;
  // thread num in block
  constexpr int block_thread_num = THREADS;
  // vector id in block
  const int block_vector_id = block_thread_id / VEC_SIZE;
  // vector num in block
  constexpr int block_vector_num = THREADS / VEC_SIZE;
  // thread id in vector
  const int vector_thread_id = block_thread_id % VEC_SIZE;
  // thread num in vector
  constexpr int vector_thread_num = VEC_SIZE;

  __shared__ T block_sum;

  if (block_vector_id == 0) {
    if (vector_thread_id == 0) {
      block_sum = static_cast<T>(0);
    }
    const int rounds = len / VEC_SIZE + (len % VEC_SIZE == 0 ? 0 : 1);
    for (int i = 0; i < rounds; ++i) {
      const int idx = i * VEC_SIZE + vector_thread_id;
      // wavefront reduce
      T sum = idx < len ? buffer[idx] : 0;
#pragma unroll
      for (int j = 1; j < VEC_SIZE; j <<= 1) {
        T tmp = __shfl_up(sum, j, VEC_SIZE);
        if (vector_thread_id >= j) {
          sum += tmp;
        }
      }
      if (idx < len) {
        buffer[idx] = block_sum + sum;
      }
      if (i == rounds - 1) {
        if (vector_thread_id == VEC_SIZE - 1) {
          block_sum = sum;
        }
      } else {
        if (vector_thread_id == len - i * VEC_SIZE) {
          block_sum = sum;
        }
      }
    }
  }
}

template <typename T, size_t R, size_t THREADS, size_t WF_SIZE>
__global__ void reduction_addend_kernel(const size_t len, T *array, T *buffer) {
  // thread id in global
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  // thread num in global
  const int global_thread_num = blockDim.x * gridDim.x;
  // block id in global
  const int global_block_id = blockIdx.x;
  // block num in global
  const int global_block_num = gridDim.x;
  // thread id in block
  const int block_thread_id = threadIdx.x;
  // thread num in block
  constexpr int block_thread_num = THREADS;

  const int process_each_block = R * THREADS;
  if (global_block_id == 0) {
    return;
  }
  const int block_scan_begin = global_block_id * process_each_block;
  const int block_scan_end = min(block_scan_begin + process_each_block, len);
  for (int i = 0; i < R; ++i) {
    const int idx = block_scan_begin + THREADS * i + block_thread_id;
    if (idx < block_scan_end) {
      array[idx] += buffer[global_block_id - 1];
    }
  }
}

#endif // SPGEMM_KOKKOS_UTILS_KERNEL_HPP
