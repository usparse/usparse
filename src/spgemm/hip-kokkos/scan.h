#ifndef SPGEMM_HIP_KOKKOS_SCAN_H
#define SPGEMM_HIP_KOKKOS_SCAN_H

#include "scan_impl.hpp"

template <typename T> void scan_in_place(const size_t len, T *array);

template <typename T> void scan_single(const size_t len, T *array);

#endif // SPGEMM_HIP_KOKKOS_SCAN_H
