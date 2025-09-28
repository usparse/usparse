//
// Created by genshen to 2025-07-09.
//

#ifndef USPARSE_KERNELS_LEVEL1_USPARSE_LEVEL1_H
#define USPARSE_KERNELS_LEVEL1_USPARSE_LEVEL1_H

namespace usparse {

  template <typename INDEX_TYPE, typename DATA_TYPE>
  void device_calculateNorm(const DATA_TYPE *d_vec, INDEX_TYPE N, DATA_TYPE *d_result, DATA_TYPE *d_temp_buffer,
                            std::size_t *temp_buffer_size_byte);

  template <typename INDEX_TYPE, typename DATA_TYPE>
  void device_dotProduct(const DATA_TYPE *d_x, const DATA_TYPE *d_y, INDEX_TYPE N, DATA_TYPE *d_result,
                         DATA_TYPE *d_temp_buffer, std::size_t *temp_buffer_size_byte);

  /**
   * fusion kernel for vector dot production and axpy:
   *   for (k = 0; k <= i; k++) {
   *     H[k][i] <- y \dot V_k  // dot prod
   *     alpha <- -H[k][i]
   *     y <- y + alpha \times V_k  // axpy
   *  }
   *
   * @param d_r0 vector y (device side).
   * @param d_v vector V (device side).
   * @param d_h vector H (device side).
   * @param d_temp_buffer temp buffer used in reduction of dot-production (device side).
   * @param N length of above vectors.
   * @param m the max restart times (currently not used). see also: RESTART_TIMES.
   * @param i the loop times (see above pseudocode).
   */
  template <typename INDEX_TYPE, typename DATA_TYPE, int OPT_LEVEL>
  void dotprod_axpy_fusion(DATA_TYPE *d_r0, const DATA_TYPE *d_v, DATA_TYPE *d_h, DATA_TYPE *d_temp_buffer,
                           const INDEX_TYPE N, const INDEX_TYPE m, const INDEX_TYPE i,
                           std::size_t *temp_buffer_size_byte);

  template <typename INDEX_TYPE, typename DATA_TYPE, int THREADS, int ELES_PER_THREAD, int STRIDES, int OPT_LEVEL>
  void dotprod_axpy_fusion_internal_imp(DATA_TYPE *d_r0, const DATA_TYPE *d_v, DATA_TYPE *d_h, DATA_TYPE *d_temp_buffer,
                                        const INDEX_TYPE N, const INDEX_TYPE m, const INDEX_TYPE i,
                                        std::size_t *temp_buffer_size_byte);

  /**
   * perform: y = y + alpha * x, where x and y are both vectors of size N.
   * @tparam INDEX_TYPE index data type.
   * @tparam DATA_TYPE data type in the vector.
   * @param alpha alpha
   * @param x input vector of size N.
   * @param y input and output vector of size N.
   * @param N the vector size.
   */
  template <typename INDEX_TYPE, typename F_TYPE, typename DATA_TYPE>
  void device_daxpy(DATA_TYPE alpha, const F_TYPE *x, DATA_TYPE *y, const INDEX_TYPE N);

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

  /**
   * copy vector of size N from dst to src. and multiple a scale.
   * dst = alpha * src.
   * @tparam INDEX_TYPE index data type.
   * @tparam DATA_TYPE data type in the vector.
   * @param src input source vector of size N.
   * @param dst output destination vector of size N.
   * @param N the vector size.
   */
  template <typename INDEX_TYPE, typename DATA_TYPE>
  void device_copy_v2(const DATA_TYPE *src, DATA_TYPE *dst, const DATA_TYPE alpha, const INDEX_TYPE N);

  /**
   * Forced conversion precision from dst to src.
   * @tparam INDEX_TYPE index data type.
   * @tparam FOREM_TYPE data type of src.
   * @tparam TO_TYPE data type of dst.
   * @param src input source vector of size N.
   * @param dst output destination vector of size N.
   * @param N the vector size.
   */
  template <typename INDEX_TYPE, typename FORM_TYPE, typename TO_TYPE>
  void device_cast(const FORM_TYPE *src, TO_TYPE *dst, const INDEX_TYPE N);

  /**
   * Set the value of src to 0.0
   * @tparam INDEX_TYPE index data type.
   * @tparam DATA_TYPE data type of src.
   * @param src input source vector of size N.
   * @param N the vector size.
   */
  template <typename INDEX_TYPE, typename DATA_TYPE>
  void device_memset(DATA_TYPE *src, const INDEX_TYPE N);

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

#endif // USPARSE_KERNELS_LEVEL1_USPARSE_LEVEL1_H
