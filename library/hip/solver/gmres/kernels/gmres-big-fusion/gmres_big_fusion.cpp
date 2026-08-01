//
// Created by genshen on 2025/7/17.
//

#ifndef USPARSE_KERNELS__GMRES_BIG_FUSION_HPP
#define USPARSE_KERNELS__GMRES_BIG_FUSION_HPP

#include "primitives/reduce/device/usparse_reduce.hpp"

namespace usparse {
  namespace gmres {

    // In this kernel, we:
    // 1: set d_V <- d_b / beta.
    // 2: set d_s <- zero.
    // Note: we set d_s[0] <- beta using memory copy outside this kernel.
    template <typename I, typename T>
    __global__ void __launch_bounds__(512)
        big_fusion_outter_itl_v0(const T *d_b, T *d_V, T *d_s, const T beta, const I N, const I S_LEN) {
      const int gid = blockDim.x * blockIdx.x + threadIdx.x;
      const T beta_recip = 1 / beta;
      if (gid < N) {
        d_V[gid] = beta_recip * d_b[gid];
      }
      if (gid < S_LEN) {
        d_s[gid] = static_cast<T>(0);
      }
    }

    template <typename I, typename T>
    void big_fusion_outter_itl_v0_wrapper(const T *d_b, T *d_V, T *d_s, const T beta, const I N, const I S_LEN) {
      constexpr int THREADS = 512;
      const int BLOCKS = N / THREADS + (N % THREADS == 0 ? 0 : 1);
      (big_fusion_outter_itl_v0<I, T>)<<<BLOCKS, THREADS>>>(d_b, d_V, d_s, beta, N, S_LEN);
    }

    template <typename I, typename T1, typename T2, int THREADS>
    __global__ void __launch_bounds__(512)
        big_fusion_outter_itl_v1_part1(const T1 *dev_y, T2 *d_r0, const T2 *d_b, T2 *d_s, T2 *d_temp_buffer, const I N,
                                       const I S_LEN) {
      const int gid = blockDim.x * blockIdx.x + threadIdx.x;
      const int bid = blockIdx.x;
      const int tid_in_block = threadIdx.x;

      T2 local_sum = static_cast<T2>(0);
      if (gid < N) {
        local_sum = static_cast<T2>(dev_y[gid]) - d_b[gid];
        d_r0[gid] = local_sum;
        local_sum = local_sum * local_sum;
      }
      if (gid < S_LEN) {
        d_s[gid] = static_cast<T2>(0);
      }
      // reduce part1: reduce in each block
      const T2 block_sum = usparse::line_enhance_block_reduce<int, T2, THREADS, usparse::USPARSE_WAVEFRONT_SIZE>(
          local_sum, tid_in_block);
      if (tid_in_block == 0) {
        d_temp_buffer[blockIdx.x] = block_sum;
      }
    }

    template <typename I, typename T, int THREADS, bool IS_FINAL>
    __global__ void __launch_bounds__(512)
        big_fusion_outter_itl_v1_part2(T *dev_result, T *d_temp_buffer, const I data_len) {
      const int tid_in_block = threadIdx.x;
      const int gid = blockDim.x * blockIdx.x + threadIdx.x;

      // reduce part2: reduce among blocks.
      T local_sum = static_cast<T>(0);

      if (IS_FINAL) { // data_len <= THREADS
        if (blockIdx.x == 0) {
          if (tid_in_block < data_len) {
            local_sum = d_temp_buffer[tid_in_block];
          }

          T final_sum = usparse::line_enhance_block_reduce<int, T, THREADS, usparse::USPARSE_WAVEFRONT_SIZE>(
              local_sum, tid_in_block);
          if (tid_in_block == 0) {
            dev_result[0] = sqrt(final_sum);
          }
        }
      } else { // data_len > THREADS
        T local_sum = static_cast<T>(0);
        if (gid < data_len) {
          local_sum = d_temp_buffer[gid];
        }
        const T block_sum = usparse::line_enhance_block_reduce<int, T, THREADS, usparse::USPARSE_WAVEFRONT_SIZE>(
            local_sum, tid_in_block);
        if (tid_in_block == 0) {
          d_temp_buffer[blockIdx.x] = block_sum;
        }
      }
    }

    template <typename I, typename T>
    __global__ void __launch_bounds__(512)
        big_fusion_outter_itl_v1_part3(const T *d_r0, T *d_V, const T *d_beta, const I N) {
      const int gid = blockDim.x * blockIdx.x + threadIdx.x;
      const T beta = d_beta[0]; // bcast it.
      const T beta_recip = 1 / beta;
      if (gid < N) {
        d_V[gid] = -d_r0[gid] * beta_recip;
      }
    }

    template <typename I, typename T1, typename T2>
    void big_fusion_outter_itl_v1(T1 *dev_y, T2 *d_r0, const T2 *d_b, T2 *d_V, T2 *d_s, T2 *d_temp_buffer, T2 *d_beta,
                                  const I N, const I S_LEN) {
      constexpr int THREADS = 512;
      const int BLOCKS = N / THREADS + (N % THREADS == 0 ? 0 : 1);
      // convert data type and perform norm (reduce inside block)
      big_fusion_outter_itl_v1_part1<uint, double, float, THREADS>
          <<<BLOCKS, THREADS>>>(dev_y, d_r0, d_b, d_s, d_temp_buffer, N, S_LEN);
      // perform norm
      if (BLOCKS <= THREADS) {
        // reduce among `BLOCKS` blocks and sqrt
        big_fusion_outter_itl_v1_part2<uint, float, THREADS, true><<<1, THREADS>>>(d_beta, d_temp_buffer, BLOCKS);
      } else {
        const int SECOND_BLOCKS = BLOCKS / THREADS + (BLOCKS % THREADS == 0 ? 0 : 1);
        // reduce among `BLOCKS` blocks
        big_fusion_outter_itl_v1_part2<uint, float, THREADS, false>
            <<<SECOND_BLOCKS, THREADS>>>(d_beta, d_temp_buffer, BLOCKS);
        // reduce among `SECOND_BLOCKS` blocks and sqrt
        big_fusion_outter_itl_v1_part2<uint, float, THREADS, true>
            <<<1, THREADS>>>(d_beta, d_temp_buffer, SECOND_BLOCKS);
      }
      // d_V = -d_r0/beta
      big_fusion_outter_itl_v1_part3<uint, float><<<BLOCKS, THREADS>>>(d_r0, d_V, d_beta, N);
    }

    template <typename I, typename ST, typename VT, typename XT, int II>
    __global__ void big_fusion_v2_batched_axpy_kernel(const ST *d_s_f, const VT *d_V_f, XT *dev_x, const I N,
                                                      const I VN) {
      // loop j in i+1:
      //   dev_x = dev_x + d_s_f[j] * d_V_f;
      // end loop
      const int gid = blockDim.x * blockIdx.x + threadIdx.x;
      if (gid < N) {
        VT x = static_cast<VT>(dev_x[gid]); // XT to VT
#pragma unroll
        for (int j = 0; j <= II; j++) {
          const ST alpha = d_s_f[j];
          x += alpha * d_V_f[VN * j + gid];
        }
        dev_x[gid] = static_cast<XT>(x);
      }
    }

    template <typename I, typename ST, typename VT, typename XT>
    void big_fusion_v2_batched_axpy(const ST *d_s_f, const VT *d_V_f, XT *dev_x, const I ii, const I N, const I VN) {
      constexpr int THREADS = 512;
      const int BLOCKS = N / THREADS + (N % THREADS == 0 ? 0 : 1);

#define SWITCH_CASE_SHORT_CODE(II)                                                                                     \
  case (II):                                                                                                           \
    big_fusion_v2_batched_axpy_kernel<I, ST, VT, XT, II><<<BLOCKS, THREADS>>>(d_s_f, d_V_f, dev_x, N, VN);             \
    break;

      switch (ii) {
        SWITCH_CASE_SHORT_CODE(0);
        SWITCH_CASE_SHORT_CODE(1);
        SWITCH_CASE_SHORT_CODE(2);
        SWITCH_CASE_SHORT_CODE(3);
        SWITCH_CASE_SHORT_CODE(4);
        SWITCH_CASE_SHORT_CODE(5);
        SWITCH_CASE_SHORT_CODE(6);
        SWITCH_CASE_SHORT_CODE(7);
        SWITCH_CASE_SHORT_CODE(8);
        SWITCH_CASE_SHORT_CODE(9);
        SWITCH_CASE_SHORT_CODE(10);
        SWITCH_CASE_SHORT_CODE(11);
        SWITCH_CASE_SHORT_CODE(12);
        SWITCH_CASE_SHORT_CODE(13);
        SWITCH_CASE_SHORT_CODE(14);
        SWITCH_CASE_SHORT_CODE(15);
        SWITCH_CASE_SHORT_CODE(16);
        SWITCH_CASE_SHORT_CODE(17);
        SWITCH_CASE_SHORT_CODE(18);
        SWITCH_CASE_SHORT_CODE(19);
        SWITCH_CASE_SHORT_CODE(20);
      default:;
      }
#undef SWITCH_CASE_SHORT_CODE
    }
  } // namespace gmres
} // namespace usparse

template void usparse::gmres::big_fusion_outter_itl_v0_wrapper<uint, float>(const float *d_b, float *d_V, float *d_s,
                                                                            const float beta, const uint N,
                                                                            const uint S_LEN);

template void usparse::gmres::big_fusion_outter_itl_v1<uint, double, float>(double *dev_y, float *d_r0,
                                                                            const float *d_b, float *d_V, float *d_s,
                                                                            float *d_temp_buffer, float *d_beta,
                                                                            const uint N, const uint S_LEN);

template void usparse::gmres::big_fusion_v2_batched_axpy<uint, float, float, double>(const float *d_s_f,
                                                                                     const float *d_V_f, double *dev_x,
                                                                                     const uint ii, const uint N,
                                                                                     const uint VN);
#endif // USPARSE_KERNELS__GMRES_BIG_FUSION_HPP
