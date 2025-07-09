//
// Created by genshen to 2025-07-09.
//

#ifndef USPARSE_KERNELS_LEVEL1_USPARSE_LEVEL1_H
#define USPARSE_KERNELS_LEVEL1_USPARSE_LEVEL1_H

namespace usparse {

  template <typename INDEX_TYPE, typename DATA_TYPE>
  void device_calculateNorm(const DATA_TYPE *d_vec, INDEX_TYPE N, DATA_TYPE *d_result);

  template <typename INDEX_TYPE, typename DATA_TYPE>
  void device_dotProduct(const DATA_TYPE *d_x, const DATA_TYPE *d_y, INDEX_TYPE N, DATA_TYPE *d_result);

  template <typename INDEX_TYPE, typename DATA_TYPE>
  void device_daxpy(DATA_TYPE alpha, const DATA_TYPE *x, DATA_TYPE *y, INDEX_TYPE N);

  template <typename INDEX_TYPE, typename DATA_TYPE> void device_dscal(DATA_TYPE alpha, DATA_TYPE *x, INDEX_TYPE N);

  template <typename INDEX_TYPE, typename DATA_TYPE>
  void device_dcopy(const DATA_TYPE *src, DATA_TYPE *dst, INDEX_TYPE N);
} // namespace usparse

#endif // USPARSE_KERNELS_LEVEL1_USPARSE_LEVEL1_H
