//
// Created by genshen on 2025/5/8.
//
// wrapper functions in usparse to call spmv-acc lib.
//

#ifndef USPARSE_HIP_LEVEL2_CSRMV_H
#define USPARSE_HIP_LEVEL2_CSRMV_H

#include "include/usparse.h"

namespace usparse {
  namespace spmv {
    template <typename I, typename T>
    usparse_status usparse_csrmv_analysis(const usparse_operation trans, const csr_desc<I, T> *host_csr_mat,
                                          const csr_desc<I, T> *dev_csr_mat, mat::usparse_mat_info *info,
                                          const usparse_csrmv_algorithms algo);

    template <typename I, typename T>
    usparse_status usparse_csrmv(const usparse_operation trans, const T *alpha, const T *beta,
                                 const csr_desc<I, T> *host_csr_mat, const csr_desc<I, T> *dev_csr_mat, const T *x,
                                 T *y, mat::usparse_mat_info *info, const usparse_csrmv_algorithms algo);

  } // namespace spmv
} // namespace usparse

#endif // USPARSE_HIP_LEVEL2_CSRMV_H
