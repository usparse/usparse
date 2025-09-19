//
// Created by genshen on 2025/05/08.
//

#ifndef USPARSE_HIP_USPARSE_H
#define USPARSE_HIP_USPARSE_H

#include "usparse_handle.h"
#include "usparse_mat_info.h"
#include "usparse_matrix.h"
#include "usparse_types.h"

namespace usparse {

  /**
   * Currently, we can only process usparse_fp32 and usparse_fp64 types.
   */
  namespace spmv {
    enum class usparse_csrmv_algorithms { Flat, LineEnhance, CsrAdaptivePlus, CsrVector, Adaptive };

    enum class usparse_coomv_algorithms { Flat, LineEnhance, CsrAdaptivePlus, CsrVector, Adaptive };

    usparse_status usparse_scsrmv_analysis(const usparse_operation trans,
                                           const csr_desc<usparse_int, usparse_fp32> dev_csr_mat,
                                           mat::usparse_mat_info *info, const usparse_csrmv_algorithms algo);

    usparse_status usparse_dcsrmv_analysis(const usparse_operation trans,
                                           const csr_desc<usparse_int, usparse_fp64> dev_csr_mat,
                                           mat::usparse_mat_info *info, const usparse_csrmv_algorithms algo);

    usparse_status usparse_scsrmv(const usparse_operation trans, const usparse_fp32 alpha, const usparse_fp32 beta,
                                  const csr_desc<usparse_int, usparse_fp32> dev_csr_mat, const usparse_fp32 *x,
                                  usparse_fp32 *y, mat::usparse_mat_info *info, const usparse_csrmv_algorithms algo);

    usparse_status usparse_dcsrmv(const usparse_operation trans, const usparse_fp64 alpha, const usparse_fp64 beta,
                                  const csr_desc<usparse_int, usparse_fp64> dev_csr_mat, const usparse_fp64 *x,
                                  usparse_fp64 *y, mat::usparse_mat_info *info, const usparse_csrmv_algorithms algo);

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
