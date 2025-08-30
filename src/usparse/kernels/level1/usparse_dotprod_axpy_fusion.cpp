//
// Created by genshen to 2025-08-11.
//

#include <iostream>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <hip/hip_cooperative_groups.h>

#include "gmres_optimization_config.h"
#include "kernels/primitives/usparse_reduce.hpp"
#include "kernels/usparse_config.hpp"
#include "usparse_level1.h"

namespace usparse {
  template <typename INDEX_TYPE, typename DATA_TYPE, int THREADS, int ELES_PER_THREAD, int STRIDES>
  __device__ __forceinline__ void dot_prod_cg(cooperative_groups::grid_group &grid, const int &g_tid,
                                              const DATA_TYPE (&reg_x)[STRIDES][ELES_PER_THREAD],
                                              const DATA_TYPE (&reg_y)[STRIDES][ELES_PER_THREAD],
                                              DATA_TYPE *d_temp_buffer, DATA_TYPE *result) {
    DATA_TYPE thread_local_sum = static_cast<DATA_TYPE>(0);
#pragma unroll
    for (int i = 0; i < STRIDES; i++) {
#pragma unroll
      for (int j = 0; j < ELES_PER_THREAD; j++) {
        // todo: check array length. because reg_x and reg_y may be dirty data.
        thread_local_sum += reg_x[i][j] * reg_y[i][j];
      }
    }

    usparse::device::reduce_using_cooperative_groups<INDEX_TYPE, DATA_TYPE, THREADS>(grid, thread_local_sum,
                                                                                     d_temp_buffer, result);
  }

  template <typename INDEX_TYPE, typename DATA_TYPE, int THREADS, int ELES_PER_THREAD, int STRIDES>
  __device__ __forceinline__ void calculate_norm_cg(cooperative_groups::grid_group &grid, const int &g_tid,
                                                    const DATA_TYPE (&reg_x)[STRIDES][ELES_PER_THREAD],
                                                    DATA_TYPE *d_temp_buffer, DATA_TYPE *result) {
    DATA_TYPE thread_local_sum = static_cast<DATA_TYPE>(0);
#pragma unroll
    for (int i = 0; i < STRIDES; i++) {
#pragma unroll
      for (int j = 0; j < ELES_PER_THREAD; j++) {
        // todo: check array length. because reg_x may be dirty data.
        thread_local_sum += reg_x[i][j] * reg_x[i][j];
      }
    }

    usparse::device::reduce_using_cooperative_groups<INDEX_TYPE, DATA_TYPE, THREADS>(grid, thread_local_sum,
                                                                                     d_temp_buffer, result);

    // calculate norm
    if (grid.thread_rank() == 0) {
      *result = sqrt(*result);
    }
  }

  /**
   * load a array in global device memory into register.
   * each thead loads @tparam ELES_PER_THREAD * @tparam STRIDES elements.
   * ___________________________________________________________________________________________________
   * | ELES_PER_THREAD elements  | ... |  ELES_PER_THREAD elements  | ...  |  ELES_PER_THREAD elements |
   * ---------------------------------------------------------------------------------------------------
   *
   * @param g_threads_num the number of total thread in the grid.
   * @param g_tid global thread id in the grid.
   * @param dest_reg the register for storing the values.
   * @param src_mem the device memory. source data.
   * @param length data length of the source data @param src_mem.
   */
  template <typename INDEX_TYPE, typename DATA_TYPE, int ELES_PER_THREAD, int STRIDES>
  __device__ __forceinline__ void load_v_into_reg(const int &g_threads_num, const int &g_tid,
                                                  DATA_TYPE (&dest_reg)[STRIDES][ELES_PER_THREAD],
                                                  const DATA_TYPE src_mem[], const INDEX_TYPE &length) {
#pragma unroll
    for (int i = 0; i < STRIDES; i++) {
      const INDEX_TYPE base_index = (g_threads_num * i + g_tid) * ELES_PER_THREAD;
// todo: optimization tips: malloc a alligned array, and move `if` here.
#pragma unroll
      for (int j = 0; j < ELES_PER_THREAD; j++) {
        const INDEX_TYPE index = base_index + j;
        if (index < length) {
          dest_reg[i][j] = src_mem[index];
        } else {
          // set register which is out of index to zero to prevent dirty data.
          dest_reg[i][j] = static_cast<DATA_TYPE>(0);
        }
      }
    }
  }

  /**
   * store data in register back to device memory.
   */
  template <typename INDEX_TYPE, typename DATA_TYPE, int ELES_PER_THREAD, int STRIDES>
  __device__ __forceinline__ void store_v_to_device(const int &g_threads_num, const int &g_tid,
                                                    const DATA_TYPE (&src_reg)[STRIDES][ELES_PER_THREAD],
                                                    DATA_TYPE dest_mem[], const INDEX_TYPE &length) {
#pragma unroll
    for (int i = 0; i < STRIDES; i++) {
      const INDEX_TYPE base_index = (g_threads_num * i + g_tid) * ELES_PER_THREAD;
// todo: optimization tips: malloc a alligned array, and move `if` here.
#pragma unroll
      for (int j = 0; j < ELES_PER_THREAD; j++) {
        const INDEX_TYPE index = base_index + j;
        if (index < length) {
          dest_mem[index] = src_reg[i][j];
        }
      }
    }
  }

  /**
   * Only for debuging.
   * Print elements loaded from the origin device memory whose origin index is in set [start_index, end_index).
   */
  template <typename INDEX_TYPE, typename DATA_TYPE, int ELES_PER_THREAD, int STRIDES>
  __device__ __forceinline__ void __DEBUG_print_reg_values(const int &g_threads_num, const int &g_tid,
                                                           const DATA_TYPE (&src_reg)[STRIDES][ELES_PER_THREAD],
                                                           const INDEX_TYPE &start_index, const INDEX_TYPE &end_index) {
#pragma unroll
    for (int i = 0; i < STRIDES; i++) {
      const INDEX_TYPE base_index = (g_threads_num * i + g_tid) * ELES_PER_THREAD;
// todo: optimization tips: malloc a alligned array, and move `if` here.
#pragma unroll
      for (int j = 0; j < ELES_PER_THREAD; j++) {
        const INDEX_TYPE index = base_index + j;
        if (index >= start_index && index < end_index) {
          printf("[REG] index: %d, val: %f.\n", index, src_reg[i][j]);
        }
      }
    }
  }

  template <typename INDEX_TYPE, typename DATA_TYPE, int ELES_PER_THREAD, int STRIDES>
  __device__ __forceinline__ void register_daxpy(DATA_TYPE (&reg_x)[STRIDES][ELES_PER_THREAD],
                                                 const DATA_TYPE (&reg_y)[STRIDES][ELES_PER_THREAD],
                                                 const DATA_TYPE alpha) {
#pragma unroll
    for (int i = 0; i < STRIDES; i++) {
#pragma unroll
      for (int j = 0; j < ELES_PER_THREAD; j++) {
        reg_x[i][j] = reg_x[i][j] + alpha * reg_y[i][j];
      }
    }
  }

  template <typename INDEX_TYPE, typename DATA_TYPE, int ELES_PER_THREAD, int STRIDES>
  __device__ __forceinline__ void register_dscal(DATA_TYPE (&reg_x)[STRIDES][ELES_PER_THREAD], const DATA_TYPE alpha) {
#pragma unroll
    for (int i = 0; i < STRIDES; i++) {
#pragma unroll
      for (int j = 0; j < ELES_PER_THREAD; j++) {
        reg_x[i][j] = alpha * reg_x[i][j];
      }
    }
  }

  constexpr int RESTART_TIMES = 20; // todo:

  template <typename INDEX_TYPE, typename DATA_TYPE, int THREADS, int ELES_PER_THREAD, int STRIDES, int OPT_LEVEL>
  __global__ void __launch_bounds__(512)
      dotprod_axpy_fusion_kernel(DATA_TYPE *d_r0, DATA_TYPE *d_v, DATA_TYPE *d_h, DATA_TYPE *d_temp_buffer,
                                 const INDEX_TYPE N, const INDEX_TYPE m, const INDEX_TYPE i) {
    // init cooperative groups
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    const int gid = blockDim.x * blockIdx.x + threadIdx.x;
    const int g_threads_num = THREADS * gridDim.x;
    const int wf_thread_id = threadIdx.x % usparse::USPARSE_WAVEFRONT_SIZE;
    const int idx_per_block = (N / gridDim.x) + (N % gridDim.x == 0 ? 0 : 1);
    const int block_idx_begin = blockIdx.x * idx_per_block;
    const int block_idx_end = min(N, block_idx_begin + idx_per_block);

    DATA_TYPE reg_x[STRIDES][ELES_PER_THREAD];
    DATA_TYPE reg_y[STRIDES][ELES_PER_THREAD];
    // load d_r0 into reg x:
    load_v_into_reg(g_threads_num, gid, reg_x, d_r0, N);

    for (int k = 0; k <= i; k++) {
      const DATA_TYPE *d_vk = d_v + k * N;
      // load d_v[k] into register reg_y
      load_v_into_reg<INDEX_TYPE, DATA_TYPE, ELES_PER_THREAD, STRIDES>(g_threads_num, gid, reg_y, d_vk, N);

      const INDEX_TYPE h_index = k * RESTART_TIMES + i; // todo: RESTART_TIMES == m
      // dot product using cg
      dot_prod_cg<INDEX_TYPE, DATA_TYPE, THREADS, ELES_PER_THREAD, STRIDES>(grid, gid, reg_x, reg_y, d_temp_buffer,
                                                                            &(d_h[h_index]));
      grid.sync();

      const DATA_TYPE alpha = -d_h[h_index];

      // register level daxpy
      register_daxpy<INDEX_TYPE, DATA_TYPE, ELES_PER_THREAD, STRIDES>(reg_x, reg_y, alpha);
    }

    if (OPT_LEVEL == usparse::GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION_LEVEL_1) {
      // store back to d_r0;
      store_v_to_device<INDEX_TYPE, DATA_TYPE, ELES_PER_THREAD, STRIDES>(g_threads_num, gid, reg_x, d_r0, N);
    } else { // GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION_LEVEL_2
      const INDEX_TYPE h_idx = (i + 1) * RESTART_TIMES + i;
      calculate_norm_cg<INDEX_TYPE, DATA_TYPE, THREADS, ELES_PER_THREAD, STRIDES>(grid, gid, reg_x, d_temp_buffer,
                                                                                  &(d_h[h_idx]));

      grid.sync();
      const DATA_TYPE alpha = 1 / d_h[h_idx];

      register_dscal<INDEX_TYPE, DATA_TYPE, ELES_PER_THREAD, STRIDES>(reg_x, alpha);

      // store back to array d_v[i+1]
      // d_r0: The d_r0 array is not used after the current loop, just need to store d_v[i+1].
      store_v_to_device<INDEX_TYPE, DATA_TYPE, ELES_PER_THREAD, STRIDES>(g_threads_num, gid, reg_x, d_v + (i + 1) * N,
                                                                         N);
    }
  }
} // namespace usparse

template <typename INDEX_TYPE, typename DATA_TYPE, int THREADS, int ELES_PER_THREAD, int STRIDES, int OPT_LEVEL>
void usparse::dotprod_axpy_fusion_internal_imp(DATA_TYPE *d_r0, const DATA_TYPE *d_v, DATA_TYPE *d_h,
                                               DATA_TYPE *d_temp_buffer, const INDEX_TYPE N, const INDEX_TYPE m,
                                               const INDEX_TYPE i, std::size_t *temp_buffer_size_byte) {
  constexpr int ELES_PER_BLOCK = THREADS * ELES_PER_THREAD * STRIDES;
  INDEX_TYPE BLOCKS = N / ELES_PER_BLOCK + (N % ELES_PER_BLOCK == 0 ? 0 : 1);

  if (d_temp_buffer == nullptr) {
    *temp_buffer_size_byte = BLOCKS * sizeof(DATA_TYPE);
    return;
  } else {
    void *kernelArgs[] = {(void *)&d_r0, (void *)&d_v, (void *)&d_h, (void *)&d_temp_buffer,
                          (void *)&N,    (void *)&m,   (void *)&i};
    hipError_t launchErr = hipLaunchCooperativeKernel(
        dotprod_axpy_fusion_kernel<INDEX_TYPE, DATA_TYPE, THREADS, ELES_PER_THREAD, STRIDES, OPT_LEVEL>, BLOCKS,
        THREADS, kernelArgs, 0, 0);

    if (launchErr != hipSuccess) {
      std::cerr << "Kernel launch failed: " << hipGetErrorString(launchErr) << std::endl;
    }
  }
}

template <typename INDEX_TYPE, typename DATA_TYPE, int OPT_LEVEL>
void usparse::dotprod_axpy_fusion(DATA_TYPE *d_r0, const DATA_TYPE *d_v, DATA_TYPE *d_h, DATA_TYPE *d_temp_buffer,
                                  const INDEX_TYPE N, const INDEX_TYPE m, const INDEX_TYPE i,
                                  std::size_t *temp_buffer_size_byte) {

#define FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES)                                                           \
  dotprod_axpy_fusion_internal_imp<INDEX_TYPE, DATA_TYPE, THREADS, ELES_PER_THREAD, STRIDES, OPT_LEVEL>(               \
      d_r0, d_v, d_h, d_temp_buffer, N, m, i, temp_buffer_size_byte);

  if (N < 163840) { // 163840
    constexpr int THREADS = 512;
    constexpr int ELES_PER_THREAD = 4;
    constexpr int STRIDES = 2;
    FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES);
  } else if (N < 1000000) {
    constexpr int THREADS = 512;
    constexpr int ELES_PER_THREAD = 4;
    constexpr int STRIDES = 4;
    FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES);
  } else {
    constexpr int THREADS = 512;
    constexpr int ELES_PER_THREAD = 4;
    constexpr int STRIDES = 4;
    FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES);
  }

#undef FUSION_SHORT_MARK
}

#define FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES)                                                           \
  template void usparse::dotprod_axpy_fusion_internal_imp<uint, float, THREADS, ELES_PER_THREAD, STRIDES,              \
                                                          usparse::GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION_LEVEL_2>(    \
      float *, const float *, float *, float *, const uint, const uint, const uint, std::size_t *);

FUSION_SHORT_MARK(128, 2, 2);
FUSION_SHORT_MARK(128, 2, 4);
FUSION_SHORT_MARK(128, 4, 2);
FUSION_SHORT_MARK(128, 4, 4);

FUSION_SHORT_MARK(256, 2, 2);
FUSION_SHORT_MARK(256, 2, 4);
FUSION_SHORT_MARK(256, 4, 2);
FUSION_SHORT_MARK(256, 4, 4);

FUSION_SHORT_MARK(512, 2, 2);
FUSION_SHORT_MARK(512, 2, 4);
FUSION_SHORT_MARK(512, 4, 2);
FUSION_SHORT_MARK(512, 4, 4);

#undef FUSION_SHORT_MARK

template void usparse::dotprod_axpy_fusion<uint, double, usparse::GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION_LEVEL_1>(
    double *d_r0, const double *d_v, double *d_h, double *d_temp_buffer, const uint N, const uint m, const uint i,
    std::size_t *temp_buffer_size_byte);

template void usparse::dotprod_axpy_fusion<uint, float, usparse::GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION_LEVEL_1>(
    float *d_r0, const float *d_v, float *d_h, float *d_temp_buffer, const uint N, const uint m, const uint i,
    std::size_t *temp_buffer_size_byte);

template void usparse::dotprod_axpy_fusion<uint, double, usparse::GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION_LEVEL_2>(
    double *d_r0, const double *d_v, double *d_h, double *d_temp_buffer, const uint N, const uint m, const uint i,
    std::size_t *temp_buffer_size_byte);

template void usparse::dotprod_axpy_fusion<uint, float, usparse::GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION_LEVEL_2>(
    float *d_r0, const float *d_v, float *d_h, float *d_temp_buffer, const uint N, const uint m, const uint i,
    std::size_t *temp_buffer_size_byte);
