#ifndef SPGEMM_HIP_KOKKOS_SCAN_IMPL_HPP
#define SPGEMM_HIP_KOKKOS_SCAN_IMPL_HPP

#include "error_macro.hpp"
#include "common/scan_kernel.hpp"

template <typename T> size_t scan_in_place_buffer_size(const size_t len) {
  constexpr size_t THREADS = 256;
  constexpr size_t R = 4;
  const size_t BLOCKS = len / (THREADS * R) + (len % (THREADS * R) == 0 ? 0 : 1);
  const size_t BUFFER_SIZE = (BLOCKS) * sizeof(T);
  return BUFFER_SIZE * 2;
}

template <typename T> void scan_in_place(const size_t len, T *array, T *temp_buffer) {
  constexpr size_t THREADS = 256;
  constexpr size_t VEC_SIZE = 32;
  constexpr size_t WF_SIZE = 64;
  constexpr size_t R = 4;
  constexpr size_t WF_VEC_NUM = THREADS / VEC_SIZE;
  if (len < VEC_SIZE) {
    (buffer_reduce_kernel<T, VEC_SIZE, THREADS, WF_SIZE>)<<<1, THREADS>>>(len, array);
  } else {
    const size_t BLOCKS = len / (THREADS * R) + (len % (THREADS * R) == 0 ? 0 : 1);
    (prefix_sum_kernel<T, VEC_SIZE, R, THREADS, WF_SIZE>)<<<BLOCKS, THREADS>>>(len, array, temp_buffer);
    scan_in_place(BLOCKS, temp_buffer, temp_buffer + BLOCKS);
    (reduction_addend_kernel<T, R, THREADS, WF_SIZE>)<<<BLOCKS, THREADS>>>(len, array, temp_buffer);
  }
}

template <typename T> void scan_single(const size_t len, T *array) { (scan_single_kernel<T>)<<<1, 512>>>(len, array); }

#endif // SPGEMM_HIP_KOKKOS_SCAN_IMPL_HPP
