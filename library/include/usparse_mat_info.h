//
// Created by genshen on 2025/09/19.
//

#ifndef USPARSE_HIP_USPARSE_MTX_INFO_H
#define USPARSE_HIP_USPARSE_MTX_INFO_H

#include "../hip/level2/coomv_kernels/coo_mtx_analysis_info.h"
#include "../hip/level2/spmv-acc-csrmv/hip-flat/flat_mtx_info.h"
#include "usparse_types.h"

namespace usparse {
  namespace mat {
    struct csrmv_mat_info {
      csrmv_flat_mtx_info *flat_info;
    };

    // mat info stores the information that is gathered during the analysis stage.
    struct usparse_mat_info {
    // protected:
      csrmv_mat_info *csrmv_info = nullptr;
      usparse::spmv::internal::coomv_conversion_mtx_info<usparse::usparse_int> *coomv_info = nullptr;
    };

    usparse_status usparse_create_mat_info(usparse_mat_info *info);

    usparse_status usparse_destroy_mat_info(usparse_mat_info *info);
  } // namespace mat
} // namespace usparse

#endif // USPARSE_HIP_USPARSE_MTX_INFO_H
