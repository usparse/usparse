//
// Created by genshen to 2025-07-09.
//

#ifndef USPARSE_KERNELS_LEVEL2_USPARSE_CSRMV_H
#define USPARSE_KERNELS_LEVEL2_USPARSE_CSRMV_H

#include "../spmv/api/types.h"
#include "../spmv/hip-flat/flat_config.h"
#include "../spmv/hip-flat/spmv_hip_acc_imp.h"
#include <hipsparse/hipsparse.h>

namespace usparse {

  struct hipsparse_spmv_info {
    hipsparseHandle_t handle;
    hipsparseMatDescr_t descrA;
  };

  struct hipsparse_csr_adaptive_info {
    rocsparse_handle handle;
    rocsparse_mat_descr descrA;
    rocsparse_mat_info info;
  };

  struct csrmv_mat_info {
    csrmv_flat_mtx_info flat_info;
    hipsparse_spmv_info hipsparse_default_info;
    hipsparse_csr_adaptive_info hipsparse_csr_adaptive_info;
  };

  using usparse_csrmv_algo = int;
  constexpr usparse_csrmv_algo usparse_csrmv_algo_default = 0;
  constexpr usparse_csrmv_algo usparse_csrmv_algo_line_enhance = 1;
  constexpr usparse_csrmv_algo usparse_csrmv_algo_flat = 2;
  constexpr usparse_csrmv_algo usparse_csrmv_algo_vector = 3;
  constexpr usparse_csrmv_algo usparse_csrmv_algo_thread_row = 4;
  constexpr usparse_csrmv_algo usparse_csrmv_algo_hipsparse_default = 5;      // call hipsparseScsrmv directly.
  constexpr usparse_csrmv_algo usparse_csrmv_algo_hipsparse_csr_adaptive = 6; // call another api.
  constexpr usparse_csrmv_algo usparse_csrmv_algo_hipsparse_csr_vector = 7;   // call csr-vector api.

  template <typename INDEX_TYPE, typename VALUE_TYPE>
  void device_adaptive_analysis(int trans, const VALUE_TYPE alpha, const VALUE_TYPE beta,
                                const csr_desc<INDEX_TYPE, VALUE_TYPE> d_csr_desc, const usparse_csrmv_algo algo,
                                csrmv_mat_info *info = nullptr);

  template <typename INDEX_TYPE, typename VALUE_TYPE, int BEAT_Y_POLICY = BEAT_Y_POLICY_AUTO>
  void device_adaptive_spmv(int trans, const VALUE_TYPE alpha, const VALUE_TYPE beta,
                            const csr_desc<INDEX_TYPE, VALUE_TYPE> d_csr_desc, const VALUE_TYPE *dx, VALUE_TYPE *dy,
                            const usparse_csrmv_algo algo, csrmv_mat_info *info = nullptr);

} // namespace usparse

#endif // USPARSE_KERNELS_LEVEL2_USPARSE_CSRMV_H
