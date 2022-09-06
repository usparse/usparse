#ifndef SPGEMM_HIP_KOKKOS_MAX_NUM_H
#define SPGEMM_HIP_KOKKOS_MAX_NUM_H

#include "max_num_impl.hpp"

template <typename T> size_t max_num_buffer_size(const size_t len);

template <typename T> void max_num_in_array(const size_t len, const T *array, T *max_num, T *temp_buffer);

template <typename T> void max_num_in_array_single_thread(const size_t len, const T *array, T *max_num);

#endif // SPGEMM_HIP_KOKKOS_MAX_NUM_H
