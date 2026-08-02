//
// Created by genshen on 2025/7/17.
//

#ifndef USPARSE_KERNELS__GMRES_BIG_FUSION_HPP
#define USPARSE_KERNELS__GMRES_BIG_FUSION_HPP

namespace usparse {
  namespace gmres {

    template <typename I, typename T>
    void big_fusion_outter_itl_v0_wrapper(const T *d_b, T *d_V, T *d_s, const T beta, const I N, const I S_LEN);

    template <typename I, typename T1, typename T2>
    void big_fusion_outter_itl_v1(T1 *dev_y, T2 *d_r0, const T2 *d_b, T2 *d_V, T2 *d_s, T2 *d_temp_buffer, T2 *d_beta,
                                  const I N, const I S_LEN);

    template <typename I, typename ST, typename VT, typename XT>
    void big_fusion_v2_batched_axpy(const ST *d_s_f, const VT *d_V_f, XT *dev_x, const I ii, const I N, const I VN);
  } // namespace gmres
} // namespace usparse

#endif // USPARSE_KERNELS__GMRES_BIG_FUSION_HPP
