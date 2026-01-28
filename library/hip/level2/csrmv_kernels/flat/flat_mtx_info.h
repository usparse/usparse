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
};

using usparse_default_flat_mtx_info = csrmv_flat_mtx_info<usparse::usparse_int, usparse::usparse_fp64>;

#endif // USPARSE_HIP_USPARSE_FLAT_MTX_INFO_H
