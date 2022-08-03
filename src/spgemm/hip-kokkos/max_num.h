#ifndef SPGEMM_HIP_KOKKOS_MAX_NUM_H
#define SPGEMM_HIP_KOKKOS_MAX_NUM_H

#include "max_num_impl.hpp"

template <typename T> void max_num_in_array(const size_t len, const T *array, T *max_num);

template <typename T> void max_num_in_array_single_thread(const size_t len, const T *array, T *max_num);

#endif // SPGEMM_HIP_KOKKOS_MAX_NUM_H
