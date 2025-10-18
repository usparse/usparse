//
// Created by genshen on 2025/09/28.
//

#ifndef USPARSE_HIP_USPARSE_LEVEL2_COOMV_KERNELS_COO_MTX_ANALYSIS_H
#define USPARSE_HIP_USPARSE_LEVEL2_COOMV_KERNELS_COO_MTX_ANALYSIS_H

#include "../csrmv_kernels/flat/flat_mtx_info.h"

namespace usparse {

  namespace spmv {
    namespace internal {
      /**
       * the matrix info for all conversion-based coomv methods.
       */
      template <typename I> struct coomv_conversion_mtx_info {
        I *csr_row_ptr = nullptr;
        csrmv_flat_mtx_info *flat_info = nullptr; // the flat analysis result.
      };
    } // namespace internal
  } // namespace spmv
} // namespace usparse

#endif // USPARSE_HIP_USPARSE_LEVEL2_COOMV_KERNELS_COO_MTX_ANALYSIS_H
