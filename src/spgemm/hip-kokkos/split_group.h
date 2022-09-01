#ifndef SPGEMM_HIP_KOKKOS_SPLIT_GROUP_H
#define SPGEMM_HIP_KOKKOS_SPLIT_GROUP_H

#include "split_group_impl.hpp"

template <typename I> size_t split_group_row_size(const size_t rows);

template <typename I> size_t split_group_buffer_size(const size_t rows);

template <typename I>
void split_group(const size_t rows, const I *row_ptr, I *group_row_ptr, int *hgroup_num_ptr, I *temp_buffer);

#endif // SPGEMM_HIP_KOKKOS_SPLIT_GROUP_H
