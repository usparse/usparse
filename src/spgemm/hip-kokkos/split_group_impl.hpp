#ifndef SPGEMM_HIP_KOKKOS_SPLIT_GROUP_IMPL_HPP
#define SPGEMM_HIP_KOKKOS_SPLIT_GROUP_IMPL_HPP

#include "../common/check/hip_check.hpp"
#include "../common/scan/scan.h"
#include "split_config.h"
#include "split_group_kernel.hpp"
#include <iostream>

template <typename I> size_t split_group_row_size(const size_t rows) { return rows * GROUP_NUM * sizeof(I); }

template <typename I> size_t split_group_buffer_size(const size_t rows) {
  constexpr size_t THREADS = 256;
  constexpr size_t R = 2;
  const size_t BLOCKS = rows / (THREADS * R) + (rows % (THREADS * R) == 0 ? 0 : 1);
  const size_t BUFFER_SIZE = BLOCKS * GROUP_NUM * sizeof(int);
  return BUFFER_SIZE;
}

template <typename I>
void split_group(const size_t rows, const I *row_ptr, I *group_row_ptr, int *hgroup_num_ptr, I *temp_buffer) {
  constexpr size_t THREADS = 256;
  constexpr size_t R = 2;
  const size_t BLOCKS = rows / (THREADS * R) + (rows % (THREADS * R) == 0 ? 0 : 1);
  const size_t BUFFER_SIZE = BLOCKS * GROUP_NUM * sizeof(int);

  // part 1
  (split_group_part1_kernel<I, GROUP_NUM, GROUP_THRESHOLD, R, THREADS>)<<<BLOCKS, THREADS>>>(rows, row_ptr,
                                                                                             temp_buffer);
  // scan
  I *scan_temp_buffer = nullptr;
  const size_t scan_temp_buffer_size = common::scan_buffer_size<I>(BLOCKS);
  HIP_CHECK(hipMalloc((void **)&scan_temp_buffer, scan_temp_buffer_size))

#pragma unroll GROUP_NUM
  for (int i = 0; i < GROUP_NUM; i++) {
    common::scan<I>(BLOCKS, &temp_buffer[BLOCKS * i], scan_temp_buffer);
  }
  HIP_CHECK(hipFree(scan_temp_buffer))

  for (int i = 0; i < GROUP_NUM; i++) {
    HIP_CHECK(hipMemcpy(&hgroup_num_ptr[i], &temp_buffer[BLOCKS * i + BLOCKS - 1], sizeof(int), hipMemcpyDeviceToHost))
  }

  // part 2
  (split_group_part2_kernel<I, GROUP_NUM, GROUP_THRESHOLD, R, THREADS>)<<<BLOCKS, THREADS>>>(
      rows, row_ptr, group_row_ptr, temp_buffer);
}

#endif // SPGEMM_HIP_KOKKOS_SPLIT_GROUP_IMPL_HPP
