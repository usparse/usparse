#ifndef SPGEMM_KOKKOS_UTILS_KERNEL_HPP
#define SPGEMM_KOKKOS_UTILS_KERNEL_HPP 

template<typename T, size_t BUFFER_SIZE, size_t WF_SIZE>
__global__ void continuous_acc_kernel(const size_t len, T* array) {
  // todo: optimize performance
  // thread id in global
	const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  if (global_thread_id == 0) {
    for (int I = 1; I < len; ++I) {
      array[I] += array[I - 1];
    }
  }
}

#endif //SPGEMM_KOKKOS_UTILS_KERNEL_HPP