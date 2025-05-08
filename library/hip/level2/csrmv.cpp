//
// Created by genshen on 2025/5/8.
//
// wrapper functions in usparse to call spmv-acc lib.
//

#include "csrmv.h"

template <typename I, typename T>
usparse::usparse_status usparse::spmv::usparse_csrmv_analysis(const usparse::usparse_operation trans,
                                                              const usparse::csr_desc<I, T> *host_csr_mat,
                                                              const usparse::csr_desc<I, T> *dev_csr_mat,
                                                              usparse::mat::usparse_mat_info *info,
                                                              const usparse::spmv::usparse_csrmv_algorithms algo) {
  // todo:
  return usparse::usparse_status::usparse_status_success;
}

template <typename I, typename T>
usparse::usparse_status usparse::spmv::usparse_csrmv(const usparse::usparse_operation trans, const T *alpha,
                                                     const T *beta, const usparse::csr_desc<I, T> *host_csr_mat,
                                                     const usparse::csr_desc<I, T> *dev_csr_mat, const T *x, T *y,
                                                     usparse::mat::usparse_mat_info *info,
                                                     const usparse::spmv::usparse_csrmv_algorithms algo) {

  // todo:
  return usparse::usparse_status::usparse_status_success;
}
