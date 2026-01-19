//
// Created by genshen on 2025/09/19.
//

#ifndef USPARSE_HIP_USPARSE_FLAT_MTX_INFO_H
#define USPARSE_HIP_USPARSE_FLAT_MTX_INFO_H

#include "usparse_types.h"

template <typename INDEX_TYPE>
struct csrmv_flat_mtx_info {
  INDEX_TYPE *break_points = nullptr;
};

using usparse_default_flat_mtx_info = csrmv_flat_mtx_info<usparse::usparse_int>;

#endif // USPARSE_HIP_USPARSE_FLAT_MTX_INFO_H
