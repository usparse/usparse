/**
 * Created by genshen on 2025/10/19.
 *
 */

#ifndef USPARSE_HIP_LEVEL2_COOMM_CONVERSION_BESED_H
#define USPARSE_HIP_LEVEL2_COOMM_CONVERSION_BESED_H

#include "coomm_mtx_analysis_info.h"
#include "include/usparse.h"

/**
 * In this file, we call the conversion and use the method in csrmm for coomm calculation.
 */
namespace usparse {
  namespace spmm {
    namespace internal {
      template <typename INDEX_TYPE, typename DATA_TYPE>
      void coomm_conv_analysis(const usparse::usparse_operation trans,
                               const usparse::coo_desc<INDEX_TYPE, DATA_TYPE> d_coo_desc,
                               usparse::spmm::internal::coomm_conversion_mtx_info<INDEX_TYPE> *info,
                               const usparse::spmm::usparse_coomm_algorithms algo);

      template <typename INDEX_TYPE, typename DATA_TYPE>
      void coomm_conv_compute(const usparse::usparse_operation A_trans, const usparse::usparse_operation B_trans,
                              const DATA_TYPE alpha, const DATA_TYPE beta,
                              const usparse::coo_desc<INDEX_TYPE, DATA_TYPE> A_coo, const INDEX_TYPE B_cols,
                              const DATA_TYPE *B_dense_val, DATA_TYPE *C_dense_val,
                              usparse::spmm::internal::coomm_conversion_mtx_info<INDEX_TYPE> *info,
                              const usparse::spmm::usparse_coomm_algorithms algo);

      template <typename INDEX_TYPE>
      void coomm_conv_clear(usparse::spmm::internal::coomm_conversion_mtx_info<INDEX_TYPE> *info);

    } // namespace internal
  } // namespace spmm
} // namespace usparse

#endif // USPARSE_HIP_LEVEL2_COOMM_CONVERSION_BESED_H
