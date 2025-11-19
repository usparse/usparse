//
// Created by genshen to 2025-07-09.
//

#ifndef USPARSE_KERNELS_LEVEL1_USPARSE_LEVEL1_H
#define USPARSE_KERNELS_LEVEL1_USPARSE_LEVEL1_H

#include "usparse_level1_batched.h"

namespace usparse {
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
                           const INDEX_TYPE N, const INDEX_TYPE VN, const INDEX_TYPE m, const INDEX_TYPE i,
                           std::size_t *temp_buffer_size_byte, const float f_density);

  template <typename INDEX_TYPE, typename DATA_TYPE, int THREADS, int ELES_PER_THREAD, int STRIDES, int OPT_LEVEL>
  void dotprod_axpy_fusion_internal_imp(DATA_TYPE *d_r0, const DATA_TYPE *d_v, DATA_TYPE *d_h, DATA_TYPE *d_temp_buffer,
                                        const INDEX_TYPE N, const INDEX_TYPE VN, const INDEX_TYPE m, const INDEX_TYPE i,
                                        std::size_t *temp_buffer_size_byte);


} // namespace usparse

#endif // USPARSE_KERNELS_LEVEL1_USPARSE_LEVEL1_H
