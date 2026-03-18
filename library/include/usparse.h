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

    enum class usparse_coomv_algorithms { NaiveFlat, Flat, LineEnhance, AdaptivePlus, Vector, Adaptive };

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

    usparse_status usparse_csrmv_clear(mat::usparse_mat_info *info, const usparse::spmv::usparse_csrmv_algorithms algo);

    usparse_status usparse_scoomv_analysis(const usparse_operation trans,
                                           const coo_desc<usparse_int, usparse_fp32> host_coo_mat,
                                           const coo_desc<usparse_int, usparse_fp32> dev_coo_mat,
                                           mat::usparse_mat_info *info, const usparse_coomv_algorithms algo);

    usparse_status usparse_dcoomv_analysis(const usparse_operation trans,
                                           const coo_desc<usparse_int, usparse_fp64> host_coo_mat,
                                           const coo_desc<usparse_int, usparse_fp64> dev_coo_mat,
                                           mat::usparse_mat_info *info, const usparse_coomv_algorithms algo);

    usparse_status usparse_scoomv(const usparse_operation trans, const usparse_fp32 alpha, const usparse_fp32 beta,
                                  const coo_desc<usparse_int, usparse_fp32> host_coo_mat,
                                  const coo_desc<usparse_int, usparse_fp32> dev_coo_mat, const usparse_fp32 *x,
                                  usparse_fp32 *y, mat::usparse_mat_info *info, const usparse_coomv_algorithms algo);

    usparse_status usparse_dcoomv(const usparse_operation trans, const usparse_fp64 alpha, const usparse_fp64 beta,
                                  const coo_desc<usparse_int, usparse_fp64> host_coo_mat,
                                  const coo_desc<usparse_int, usparse_fp64> dev_coo_mat, const usparse_fp64 *x,
                                  usparse_fp64 *y, mat::usparse_mat_info *info, const usparse_coomv_algorithms algo);

    usparse_status usparse_coomv_clear(mat::usparse_mat_info *info);
  } // namespace spmv

  namespace spgemm {
    // todo:
  } // namespace spgemm

  namespace spmm {
    // todo: analysis and clear api

    enum class usparse_csrmm_algorithms { Vector, Adaptive };
    enum class usparse_coomm_algorithms { Vector, Adaptive };

    /**
     * C = alpha * A * B + beta * C. where A is CSR sparse matrix, B and C are dense matrix.
     * A is m rows and k columns. B is k rows and n columns. C is m rows and n columns.
     */
    usparse_status usparse_scsrmm(const usparse_handle handle, const usparse_operation A_trans,
                                  const usparse_operation B_trans, const usparse_fp32 alpha, const usparse_fp32 beta,
                                  const csr_desc<usparse_int, usparse_fp32> dev_csr_mat_a,
                                  const usparse_fp32 *B_dense_val, const usparse_int B_cols, const usparse_int ldb,
                                  usparse_fp32 *C_dense_val, const usparse_int ldc, const usparse_csrmm_algorithms algo);

    usparse_status usparse_dcsrmm(const usparse_handle handle, const usparse_operation A_trans,
                                  const usparse_operation B_trans, const usparse_fp64 alpha, const usparse_fp64 beta,
                                  const csr_desc<usparse_int, usparse_fp64> dev_csr_mat_a,
                                  const usparse_fp64 *B_dense_val, const usparse_int B_cols, const usparse_int ldb,
                                  usparse_fp64 *C_dense_val, const usparse_int ldc, const usparse_csrmm_algorithms algo);

  } // namespace spmm

  namespace sptrsv {
    // todo:
  } // namespace sptrsv

  namespace sptrsm {
    // todo:
  } // namespace sptrsm

} // namespace usparse

#endif // USPARSE_HIP_USPARSE_H
