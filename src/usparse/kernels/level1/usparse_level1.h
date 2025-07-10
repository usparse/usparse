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

  /**
   * perform: y = y + alpha * x, where x and y are both vectors of size N.
   * @tparam INDEX_TYPE index data type.
   * @tparam DATA_TYPE data type in the vector.
   * @param alpha alpha
   * @param x input vector of size N.
   * @param y input and output vector of size N.
   * @param N the vector size.
   */
  template <typename INDEX_TYPE, typename DATA_TYPE>
  void device_daxpy(DATA_TYPE alpha, const DATA_TYPE *x, DATA_TYPE *y, const INDEX_TYPE N);

  /**
   * perform: x = alpha * x, where x is a vector of size N.
   * @tparam INDEX_TYPE index data type.
   * @tparam DATA_TYPE data type in the vector.
   * @param alpha alpha
   * @param x input and output vector of size N.
   * @param N the vector size.
   */
  template <typename INDEX_TYPE, typename DATA_TYPE>
  void device_dscal(const DATA_TYPE alpha, DATA_TYPE *x, const INDEX_TYPE N);

  /**
   * copy vector of size N from dst to src.
   * @tparam INDEX_TYPE index data type.
   * @tparam DATA_TYPE data type in the vector.
   * @param src input source vector of size N.
   * @param dst output destination vector of size N.
   * @param N the vector size.
   */
  template <typename INDEX_TYPE, typename DATA_TYPE>
  void device_dcopy(const DATA_TYPE *src, DATA_TYPE *dst, const INDEX_TYPE N);
} // namespace usparse

#endif // USPARSE_KERNELS_LEVEL1_USPARSE_LEVEL1_H
