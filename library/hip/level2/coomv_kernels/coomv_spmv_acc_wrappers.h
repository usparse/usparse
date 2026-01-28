/**
 * Created by genshen on 2025/9/28.
 *
 */

#ifndef USPARSE_HIP_LEVEL2_COOMV_SPMV_ACC_WRAPPERS_H
#define USPARSE_HIP_LEVEL2_COOMV_SPMV_ACC_WRAPPERS_H

#include "../spmv_api_config.hpp"
#include "coo_mtx_analysis_info.h"
#include "include/usparse.h"

/**
 * In this file, we call the adaptive/flat/line-enhance/vector methods in csrmv for coomv calculation.
 */
namespace usparse {
  namespace spmv {
    namespace internal {
      template <typename INDEX_TYPE, typename DATA_TYPE>
      void coomv_conv_analysis(const usparse::usparse_operation trans,
                               const usparse::coo_desc<INDEX_TYPE, DATA_TYPE> d_coo_desc,
                               coomv_conversion_mtx_info<INDEX_TYPE, INDEX_TYPE, DATA_TYPE> *info,
                               const usparse::spmv::usparse_coomv_algorithms algo);

      template <typename INDEX_TYPE, typename DATA_TYPE>
      void coomv_conv_calculate(const usparse::usparse_operation trans, const DATA_TYPE alpha, const DATA_TYPE beta,
                                const usparse::coo_desc<INDEX_TYPE, DATA_TYPE> d_coo_desc, const DATA_TYPE *d_x,
                                DATA_TYPE *d_y, coomv_conversion_mtx_info<INDEX_TYPE, INDEX_TYPE, DATA_TYPE> *info,
                                const usparse::spmv::usparse_coomv_algorithms algo);

      template <typename INDEX_TYPE, typename DATA_TYPE>
      void coomv_conv_clear(coomv_conversion_mtx_info<INDEX_TYPE, INDEX_TYPE, DATA_TYPE> *info);

    } // namespace internal
  } // namespace spmv
} // namespace usparse

#endif // USPARSE_HIP_LEVEL2_COOMV_SPMV_ACC_WRAPPERS_H
