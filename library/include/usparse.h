//
// Created by genshen on 2025/05/08.
//

#ifndef USPARSE_HIP_USPARSE_H
#define USPARSE_HIP_USPARSE_H

#include "usparse_matrix.h"
#include "usparse_types.h"

namespace usparse {

  namespace mat {
    struct usparse_mat_info {
      usparse_int *info; // todo.
    };

    usparse_status usparse_create_mat_info(usparse_mat_info *info);

    usparse_status usparse_destroy_mat_info(usparse_mat_info *info);
  } // namespace mat

  /**
   * Currently, we can only process float and double types.
   */
  namespace spmv {
    enum class usparse_csrmv_algorithms { Flat, LineEnhance, CsrAdaptivePlus, CsrVector, Adaptive };

    usparse_status usparse_scsrmv_analysis(const usparse_operation trans,
                                           const csr_desc<usparse_int, float> *host_csr_mat,
                                           const csr_desc<usparse_int, float> *dev_csr_mat, mat::usparse_mat_info *info,
                                           const usparse_csrmv_algorithms algo);

    usparse_status usparse_dcsrmv_analysis(const usparse_operation trans,
                                           const csr_desc<usparse_int, double> *host_csr_mat,
                                           const csr_desc<usparse_int, double> *dev_csr_mat,
                                           mat::usparse_mat_info *info, const usparse_csrmv_algorithms algo);

    usparse_status usparse_scsrmv(const usparse_operation trans, const float *alpha, const float *beta,
                                  const csr_desc<usparse_int, float> *host_csr_mat,
                                  const csr_desc<usparse_int, float> *dev_csr_mat, const float *x, float *y,
                                  mat::usparse_mat_info *info, const usparse_csrmv_algorithms algo);

    usparse_status usparse_dcsrmv(const usparse_operation trans, const double *alpha, const double *beta,
                                  const csr_desc<usparse_int, double> *host_csr_mat,
                                  const csr_desc<usparse_int, double> *dev_csr_mat, const double *x, double *y,
                                  mat::usparse_mat_info *info, const usparse_csrmv_algorithms algo);

    usparse_status usparse_csrmv_clear(mat::usparse_mat_info *info);
  } // namespace spmv

  namespace spgemm {
    // todo:
  } // namespace spgemm

  namespace spmm {
    // todo:
  } // namespace spmm

  namespace sptrsv {
    // todo:
  } // namespace sptrsv

  namespace sptrsm {
    // todo:
  } // namespace sptrsm

} // namespace usparse

#endif // USPARSE_HIP_USPARSE_H
