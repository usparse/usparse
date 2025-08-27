//
// Created by genshen to 2025-08-28.
//

#ifndef USPARSE_KERNELS_USPARSE_BLAS_LEVEL1_H
#define USPARSE_KERNELS_USPARSE_BLAS_LEVEL1_H

namespace usparse {

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
   * set all elements in a vector to the given scale value: alpha.
   *
   * for(i = 0:N) {
   *    data[i] = alpha;
   * }
   * @tparam INDEX_TYPE data type for index
   * @tparam DATA_TYPE data type for vector value
   * @param alpha the scale: alpha
   * @param data the vector data
   * @param N the size of the vector data
   */
  template <typename INDEX_TYPE, typename DATA_TYPE>
  void device_scale_set(const DATA_TYPE alpha, DATA_TYPE *data, const INDEX_TYPE N);
} // namespace usparse

#endif // USPARSE_KERNELS_USPARSE_BLAS_LEVEL1_H
