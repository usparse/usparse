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
      template <typename J, typename I = J> struct coomv_conversion_mtx_info {
        J *csr_row_ptr = nullptr;
        csrmv_flat_mtx_info<I> *flat_info = nullptr; // the flat analysis result.
      };
    } // namespace internal
  } // namespace spmv
} // namespace usparse

#endif // USPARSE_HIP_USPARSE_LEVEL2_COOMV_KERNELS_COO_MTX_ANALYSIS_H
