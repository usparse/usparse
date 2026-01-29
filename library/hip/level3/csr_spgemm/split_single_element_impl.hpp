#ifndef USTB_SPARSE_SPGEMM_SPLIT_ONE_IMPL_HPP
#define USTB_SPARSE_SPGEMM_SPLIT_ONE_IMPL_HPP

#include "common/usparse_utils.hpp"
#include "spgemm_common/scan/scan_impl.hpp"
#include "align.h"
#include "split_config.h"
#include "split_group_kernel.hpp"


template <typename I>
void split_one_element(const size_t rows, const I *row_ptr, I *group_row_ptr, int *hgroup_num_ptr, void *temp_buffer, const size_t ROW_NNZ_BEGIN_VALUE) {
  constexpr size_t THREADS = 256;
  constexpr size_t R = 2;
  const size_t BLOCKS = rows / (THREADS * R) + (rows % (THREADS * R) == 0 ? 0 : 1);

  char *ptr = reinterpret_cast<char *>(temp_buffer);
  int *split_buffer = reinterpret_cast<int *>(ptr);
  ptr += ALIGN_256_BYTES(BLOCKS * GROUP_ONE_NUM * sizeof(int));

  int *split_num_buffer = reinterpret_cast<int *>(ptr);
  // part 1
  (split_group_part1_kernel<I, GROUP_ONE_NUM, CLASSIFICATION_GROUP_ONE_NUM, GROUP_ONE_THRESHOLD, R, THREADS>)<<<BLOCKS, THREADS>>>(rows, row_ptr,
                                                                                             split_buffer, ROW_NNZ_BEGIN_VALUE);
  // part 2 scan
  const size_t SCAN_SIZE = BLOCKS;
  const size_t STEP_WIDTH = BLOCKS;
  constexpr size_t PART2_THREADS = 1024;
  // ** notice: BLOCKS must less than 1024

  if (SCAN_SIZE <= PART2_THREADS) {
    constexpr size_t BLOCKS = GROUP_ONE_NUM;
    constexpr size_t THREADS = PART2_THREADS;
    constexpr size_t VEC_SIZE = __WF_SIZE__;
    (split_group_part2_kernel<I, VEC_SIZE, THREADS>)<<<BLOCKS, THREADS>>>(SCAN_SIZE, STEP_WIDTH, split_buffer,
                                                                          split_num_buffer);

    // USPARSE_HIP_API_CHECK(hipMemcpy(hgroup_num_ptr, split_num_buffer, sizeof(int) * GROUP_NUM, hipMemcpyDeviceToHost))
  } else {
    I *scan_temp_buffer = nullptr;
    const size_t scan_temp_buffer_size = common::scan_buffer_size<I>(BLOCKS);
    USPARSE_HIP_API_CHECK(hipMalloc((void **)&scan_temp_buffer, scan_temp_buffer_size))

#pragma unroll GROUP_ONE_NUM
    for (int i = 0; i < GROUP_ONE_NUM; i++) {
      common::scan<I>(BLOCKS, &split_buffer[BLOCKS * i], scan_temp_buffer);
    }

    USPARSE_HIP_API_CHECK(hipFree(scan_temp_buffer))
    for (int i = 0; i < GROUP_ONE_NUM; i++) {
      USPARSE_HIP_API_CHECK(
          hipMemcpy(&hgroup_num_ptr[i], &split_buffer[BLOCKS * i + BLOCKS - 1], sizeof(int), hipMemcpyDeviceToHost))
    }
  }
  // part 3
  (split_group_part3_kernel<I, GROUP_ONE_NUM, CLASSIFICATION_GROUP_ONE_NUM, GROUP_ONE_THRESHOLD, R, THREADS>)<<<BLOCKS, THREADS>>>(
      rows, row_ptr, group_row_ptr, split_buffer, ROW_NNZ_BEGIN_VALUE);
  if (SCAN_SIZE <= PART2_THREADS) {
    hipDeviceSynchronize();
    for (int i = 0; i < GROUP_ONE_NUM; ++i) {
      hgroup_num_ptr[i] = split_num_buffer[i];
    }
  }
}




#endif //USTB_SPARSE_SPGEMM_SPLIT_ONE_IMPL_HPP