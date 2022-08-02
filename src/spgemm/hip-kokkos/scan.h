#ifndef SPGEMM_HIP_KOKKOS_SCAN_H
#define SPGEMM_HIP_KOKKOS_SCAN_H
#include "scan_kernel.hpp"

template <typename T> void scan_in_place(const size_t len, T *array) {
  constexpr size_t THREADS = 256;
  constexpr size_t VEC_SIZE = 32;
  constexpr size_t WF_SIZE = 64;
  constexpr size_t R = 4;
  constexpr size_t WF_VEC_NUM = THREADS / VEC_SIZE;
  if (len < WF_VEC_NUM) {
    (buffer_reduce_kernel<T, VEC_SIZE, THREADS, WF_SIZE>)<<<1, THREADS>>>(len, array);
  } else {
    const size_t BLOCKS = len / (THREADS * R) + (len % (THREADS * R) == 0 ? 0 : 1);
    T *temp_buffer = nullptr;
    const size_t BUFFER_SIZE = (BLOCKS) * sizeof(T);
    KOKKOS_HIP_CHECK(hipMalloc((void **)&temp_buffer, BUFFER_SIZE))
    KOKKOS_HIP_CHECK(hipMemset(temp_buffer, 0, BUFFER_SIZE))
    (prefix_sum_kernel<T, VEC_SIZE, R, THREADS, WF_SIZE>)<<<BLOCKS, THREADS>>>(len, array, temp_buffer);
    scan_in_place(BLOCKS, temp_buffer);
    (reduction_addend_kernel<T, R, THREADS, WF_SIZE>)<<<BLOCKS, THREADS>>>(len, array, temp_buffer);
    KOKKOS_HIP_CHECK(hipFree(temp_buffer))
  }
}

template <typename T> void scan_single(const size_t len, T *array) {
  (scan_single_kernel<T>)<<<1, 512>>>(len, array);
}
#endif // SPGEMM_HIP_KOKKOS_SCAN_H
