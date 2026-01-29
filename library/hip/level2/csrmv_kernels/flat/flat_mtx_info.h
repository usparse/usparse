//
// Created by genshen on 2025/09/19.
//

#ifndef USPARSE_HIP_USPARSE_FLAT_MTX_INFO_H
#define USPARSE_HIP_USPARSE_FLAT_MTX_INFO_H

#include "usparse_types.h"

template <typename INDEX_TYPE, typename Y_TYPE> struct csrmv_flat_mtx_info {
  INDEX_TYPE *break_points = nullptr;
  // the indeices in vector y of the part sum for the final results on each hip block.
  INDEX_TYPE *y_part_indices = nullptr; // todo: its type should be ROW_TYPE
  // the part sum of the final results on each hip block.
  Y_TYPE *y_part_sum = nullptr;

  INDEX_TYPE *d_num_runs_out = nullptr;     // 存储唯一行号的数量
  INDEX_TYPE *d_unique_rows_out = nullptr;  // 存储去重后的行号
  Y_TYPE *d_merged_sums_out = nullptr;      // 存储去重合并后的值
  void *d_temp_storage = nullptr;           // rocprim 临时缓冲区
  size_t temp_storage_bytes = 0;            // 临时缓冲区大小
};

using usparse_default_flat_mtx_info = csrmv_flat_mtx_info<usparse::usparse_int, usparse::usparse_fp64>;

#endif // USPARSE_HIP_USPARSE_FLAT_MTX_INFO_H
