//
// Created by genshen on 2025/09/19.
//

#ifndef USPARSE_HIP_USPARSE_FLAT_MTX_INFO_H
#define USPARSE_HIP_USPARSE_FLAT_MTX_INFO_H

#include "usparse_types.h"
template <typename INDEX_TYPE, typename Y_TYPE> struct KV {
  INDEX_TYPE key;
  Y_TYPE value;
};

template <typename INDEX_TYPE, typename Y_TYPE> struct csrmv_flat_mtx_info {
  INDEX_TYPE *break_points = nullptr;

  // the key-value buffer: SoA
  // the indeices in vector y of the part sum for the final results on each hip block.
  // INDEX_TYPE *y_part_indices = nullptr; // todo: its type should be ROW_TYPE
  // the part sum of the final results on each hip block.
  // Y_TYPE *y_part_sum = nullptr;

  /**
   * the key-value buffer: AoS
   * the key-value buffer array for the final reduction, 
   * where the key is the index in vector y and the value is the part sum for that index in vector y.
   */
  KV<INDEX_TYPE, Y_TYPE> *key_val = nullptr;
};

using usparse_default_flat_mtx_info = csrmv_flat_mtx_info<usparse::usparse_int, usparse::usparse_fp64>;

#endif // USPARSE_HIP_USPARSE_FLAT_MTX_INFO_H
