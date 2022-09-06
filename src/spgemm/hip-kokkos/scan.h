#ifndef SPGEMM_HIP_KOKKOS_SCAN_H
#define SPGEMM_HIP_KOKKOS_SCAN_H

#include "scan_impl.hpp"

template <typename T> size_t scan_in_place_buffer_size(const size_t len);

template <typename T> void scan_in_place(const size_t len, T *array, T *temp_buffer);

template <typename T> void scan_single(const size_t len, T *array);

#endif // SPGEMM_HIP_KOKKOS_SCAN_H
