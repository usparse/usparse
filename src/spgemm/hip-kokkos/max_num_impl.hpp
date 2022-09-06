#ifndef SPGEMM_HIP_KOKKOS_MAX_NUM_IMPL_HPP
#define SPGEMM_HIP_KOKKOS_MAX_NUM_IMPL_HPP

#include "error_macro.hpp"
#include "max_num_kernel.hpp"

template <typename T> size_t max_num_buffer_size(const size_t len) {
  constexpr size_t THREADS = 256;
  constexpr size_t R = 4;
  const size_t BLOCKS = len / (THREADS * R) + (len % (THREADS * R) == 0 ? 0 : 1);
  const size_t BUFFER_SIZE = (BLOCKS) * sizeof(T);
  return BUFFER_SIZE * 2;
}

template <typename T> void max_num_in_array(const size_t len, const T *array, T *max_num, T *temp_buffer) {
  constexpr size_t THREADS = 256;
  constexpr size_t VEC_SIZE = 32;
  constexpr size_t R = 4;
  constexpr size_t WF_VEC_NUM = THREADS / VEC_SIZE;
  if (len < THREADS * R) {
    (max_num_vector_kernel<T, VEC_SIZE, R, THREADS>)<<<1, THREADS>>>(len, array, max_num);
  } else {
    const size_t BLOCKS = len / (THREADS * R) + (len % (THREADS * R) == 0 ? 0 : 1);
    (max_num_vector_kernel<T, VEC_SIZE, R, THREADS>)<<<BLOCKS, THREADS>>>(len, array, temp_buffer);
    max_num_in_array(BLOCKS, temp_buffer, max_num, temp_buffer + BLOCKS);
  }
}

template <typename T> void max_num_in_array_single_thread(const size_t len, const T *array, T *max_num) {
  (max_num_single_thread_kernel<T>)<<<1, 512>>>(len, array, max_num);
}
#endif // SPGEMM_HIP_KOKKOS_MAX_NUM_IMPL_HPP
