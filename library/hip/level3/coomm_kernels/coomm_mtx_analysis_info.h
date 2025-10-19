//
// Created by genshen on 2025/10/19.
//

#ifndef USPARSE_HIP_USPARSE_LEVEL2_COOMM_KERNELS_COO_MTX_ANALYSIS_H
#define USPARSE_HIP_USPARSE_LEVEL2_COOMM_KERNELS_COO_MTX_ANALYSIS_H

namespace usparse {
  namespace spmm {
    namespace internal {
      /**
       * the matrix info for all conversion-based coomm methods.
       */
      template <typename I> struct coomm_conversion_mtx_info {
        I *csr_row_ptr = nullptr;
      };
    } // namespace internal
  } // namespace spmm
} // namespace usparse

#endif // USPARSE_HIP_USPARSE_LEVEL2_COOMM_KERNELS_COO_MTX_ANALYSIS_H
