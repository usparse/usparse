//
// Created by genshen to 2025-10-08.
//

#ifndef USPARSE_KERNELS_USPARSE_LEVEL3_CSRMM_KERNElS_ROW_SPLIT_H
#define USPARSE_KERNELS_USPARSE_LEVEL3_CSRMM_KERNElS_ROW_SPLIT_H

namespace usparse {
  namespace internal {
    template <typename INDEX_TYPE, typename DATA_TYPE, int WF_SIZE>
    void adaptive_csrmm_vetor_row(const usparse::usparse_operation A_trans, const usparse::usparse_operation B_trans,
                                  const DATA_TYPE alpha, const DATA_TYPE beta,
                                  const usparse::csr_desc<INDEX_TYPE, DATA_TYPE> A_csr, const INDEX_TYPE B_cols,
                                  const DATA_TYPE *B_dense_val, DATA_TYPE *C_dense_val);
  } // namespace internal
} // namespace usparse
#endif // USPARSE_KERNELS_USPARSE_LEVEL3_CSRMM_KERNElS_ROW_SPLIT_H
