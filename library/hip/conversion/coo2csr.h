//
// Created by shenjihao on 2025/07/21.
//

#ifndef USPARSE_CONVERSION_COO2CSR_H
#define USPARSE_CONVERSION_COO2CSR_H

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

namespace usparse {
  namespace conversion {
    template <typename INDEX_TYPE, typename IN_TYPE, typename OUT_TYPE>
    void coo2csr(const INDEX_TYPE m, const INDEX_TYPE nnz, const IN_TYPE *coo_row_ind, OUT_TYPE *csr_row_ptr);
  }
} // namespace usparse

#endif // USPARSE_CONVERSION_ACC_COO2CSR_H
