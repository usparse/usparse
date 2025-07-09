//
// Created by genshen to 2025-07-09.
//

#ifndef USPARSE_KERNELS_LEVEL2_USPARSE_CSRMV_H
#define USPARSE_KERNELS_LEVEL2_USPARSE_CSRMV_H

namespace usparse {
  template <typename INDEX_TYPE, typename VALUE_TYPE>
  void device_csrmv(const INDEX_TYPE nrows, const INDEX_TYPE ncols, const INDEX_TYPE nnz, const INDEX_TYPE *csr_row_ptr,
                    const INDEX_TYPE *csr_col_index, const VALUE_TYPE *csr_values, const VALUE_TYPE *dev_x,
                    VALUE_TYPE *dev_y, rocsparse_handle *handle, rocsparse_mat_descr *descrA, rocsparse_mat_info *info);
}

#endif // USPARSE_KERNELS_LEVEL2_USPARSE_CSRMV_H
