//
// Created by genshen on 2025/05/08.
//

#ifndef USPARSE_HIP_USPARSE_INTERNAL_H
#define USPARSE_HIP_USPARSE_INTERNAL_H

#include "usparse.h"

namespace usparse {

  /**
   * Currently, we can only process usparse_fp32 and usparse_fp64 types.
   */
  namespace spmv {
    namespace internal {
      template <typename INDEX_TYPE, typename DATA_TYPE>
      usparse::usparse_status usparse_csrmv_analysis_imp(const usparse::usparse_operation trans,
                                                         const usparse::csr_desc<INDEX_TYPE, DATA_TYPE> dev_csr_mat,
                                                         usparse::mat::usparse_mat_info *info,
                                                         const usparse::spmv::usparse_csrmv_algorithms algo);

      template <typename INDEX_TYPE, typename DATA_TYPE>
      usparse::usparse_status
      usparse_csrmv_imp(const usparse::usparse_operation trans, const DATA_TYPE alpha, const DATA_TYPE beta,
                        const usparse::csr_desc<INDEX_TYPE, DATA_TYPE> dev_csr_mat, const DATA_TYPE *x, DATA_TYPE *y,
                        usparse::mat::usparse_mat_info *info, const usparse::spmv::usparse_csrmv_algorithms algo);

      usparse::usparse_status usparse_csrmv_clear_imp(usparse::mat::usparse_mat_info *info);

      template <typename INDEX_TYPE, typename DATA_TYPE>
      usparse::usparse_status usparse_coomv_analysis_imp(const usparse::usparse_operation trans,
                                                         const usparse::coo_desc<INDEX_TYPE, DATA_TYPE> dev_coo_mat,
                                                         usparse::mat::usparse_mat_info *info,
                                                         const usparse::spmv::usparse_coomv_algorithms algo);

      template <typename INDEX_TYPE, typename DATA_TYPE>
      usparse::usparse_status
      usparse_coomv_imp(const usparse::usparse_operation trans, const DATA_TYPE alpha, const DATA_TYPE beta,
                        const usparse::coo_desc<INDEX_TYPE, DATA_TYPE> dev_coo_mat, const DATA_TYPE *x, DATA_TYPE *y,
                        usparse::mat::usparse_mat_info *info, const usparse::spmv::usparse_coomv_algorithms algo);

      usparse::usparse_status usparse_coomv_clear_imp(usparse::mat::usparse_mat_info *info);
    } // namespace internal
  } // namespace spmv

} // namespace usparse

#endif // USPARSE_HIP_USPARSE_INTERNAL_H
