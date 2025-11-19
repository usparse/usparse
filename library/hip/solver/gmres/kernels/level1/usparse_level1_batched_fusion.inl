//
// Created by renyi to 2025-09-16.
//

#include <iostream>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <hip/hip_cooperative_groups.h>

#include "kernels/gmres_optimization_config.h"
#include "level1/usparse_blas_level1.h"
#include "primitives/reduce/device/usparse_reduce.hpp"
#include "include/usparse_config.hpp"

namespace usparse {

  template <typename INDEX_TYPE, typename DATA_TYPE, int THREADS, int ELES_PER_THREAD, int STRIDES, int ITERS>
  __device__ __forceinline__ void gs_dot_prod_cg(cooperative_groups::grid_group &grid, const int &g_tid,
                                                 const DATA_TYPE (&reg_x)[STRIDES][ELES_PER_THREAD],
                                                 const DATA_TYPE (&reg_y)[2][STRIDES][ELES_PER_THREAD],
                                                 DATA_TYPE *d_temp_buffer, DATA_TYPE *result, int h_index) {
    DATA_TYPE thread_local_sums[2];
    constexpr int RESTART_TIMES = 20;
#pragma unroll
    for (int k = 0; k <= 1; k++) {
      thread_local_sums[k] = static_cast<DATA_TYPE>(0);
    }
#pragma unroll
    for (int k = 0; k <= 1; k++) {
#pragma unroll
      for (int i = 0; i < STRIDES; i++) {
#pragma unroll
        for (int j = 0; j < ELES_PER_THREAD; j++) {
          // todo: check array length. because reg_x and reg_y may be dirty data.
          thread_local_sums[k] += reg_x[i][j] * reg_y[k][i][j];
        }
      }
    }
#pragma unroll
    for (int k = 0; k <= 1; k++) {
      usparse::device::reduce_using_cooperative_groups<INDEX_TYPE, DATA_TYPE, THREADS>(
          grid, thread_local_sums[k], d_temp_buffer + k * gridDim.x,
          &(result[k * RESTART_TIMES + h_index])); // k * RESTART_TIMES + i
    }
  }

  template <typename INDEX_TYPE, typename DATA_TYPE, int ELES_PER_THREAD, int STRIDES, int ITERS>
  __device__ __forceinline__ void gs_load_v_into_reg(const int &g_threads_num, const int &g_tid,
                                                     DATA_TYPE (&dest_reg)[2][STRIDES][ELES_PER_THREAD],
                                                     const DATA_TYPE src_mem[], const INDEX_TYPE &length) {
#pragma unroll
    for (int k = 0; k <= 1; k++) {
#pragma unroll
      for (int i = 0; i < STRIDES; i++) {
        const INDEX_TYPE base_index = (g_threads_num * i + g_tid) * ELES_PER_THREAD;
        // todo: optimization tips: malloc a alligned array, and move `if` here.
#pragma unroll
        for (int j = 0; j < ELES_PER_THREAD; j++) {
          const INDEX_TYPE index = base_index + j;
          if (index < length) {
            dest_reg[k][i][j] = src_mem[index + k * length];
          } else {
            // set register which is out of index to zero to prevent dirty data.
            dest_reg[k][i][j] = static_cast<DATA_TYPE>(0);
          }
        }
      }
    }
  }

  template <typename INDEX_TYPE, typename DATA_TYPE, int THREADS, int ELES_PER_THREAD, int STRIDES, int OPT_LEVEL,
            int ITERS>
  __global__ void __launch_bounds__(512)
      gs_fusion_kernel(DATA_TYPE *d_r0, DATA_TYPE *d_v, DATA_TYPE *d_h, DATA_TYPE *d_temp_buffer, const INDEX_TYPE N,
                       const INDEX_TYPE m, const INDEX_TYPE i) {
    // init cooperative groups
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    const int gid = blockDim.x * blockIdx.x + threadIdx.x;
    const int g_threads_num = THREADS * gridDim.x;
    const int wf_thread_id = threadIdx.x % usparse::USPARSE_WAVEFRONT_SIZE;
    const int idx_per_block = (N / gridDim.x) + (N % gridDim.x == 0 ? 0 : 1);
    const int block_idx_begin = blockIdx.x * idx_per_block;
    const int block_idx_end = min(N, block_idx_begin + idx_per_block);

    DATA_TYPE reg_x[STRIDES][ELES_PER_THREAD];
    DATA_TYPE reg_y[2][STRIDES][ELES_PER_THREAD];
    DATA_TYPE reg_z[STRIDES][ELES_PER_THREAD];
    // load d_r0 into reg x:
    load_v_into_reg(g_threads_num, gid, reg_x, d_r0, N);

    for (int k = 0; k < i; k += 2) {
      const DATA_TYPE *d_vk = d_v + k * N;
      // load d_v[k] into register reg_y
      gs_load_v_into_reg<INDEX_TYPE, DATA_TYPE, ELES_PER_THREAD, STRIDES, ITERS>(g_threads_num, gid, reg_y, d_vk, N);

      const INDEX_TYPE h_index = k * RESTART_TIMES + i;

      // dot product using cg
      gs_dot_prod_cg<INDEX_TYPE, DATA_TYPE, THREADS, ELES_PER_THREAD, STRIDES, ITERS>(grid, gid, reg_x, reg_y,
                                                                                      d_temp_buffer, d_h, h_index);

      grid.sync();

      const DATA_TYPE alpha = -d_h[h_index];
      register_daxpy<INDEX_TYPE, DATA_TYPE, ELES_PER_THREAD, STRIDES>(reg_x, reg_y[0], alpha);
      const DATA_TYPE alpha1 = -d_h[h_index + RESTART_TIMES];
      register_daxpy<INDEX_TYPE, DATA_TYPE, ELES_PER_THREAD, STRIDES>(reg_x, reg_y[1], alpha1);
    }

    if (i % 2 == 0) {
      const DATA_TYPE *d_vk = d_v + i * N;
      // load d_v[k] into register reg_y
      load_v_into_reg<INDEX_TYPE, DATA_TYPE, ELES_PER_THREAD, STRIDES>(g_threads_num, gid, reg_z, d_vk, N);

      const INDEX_TYPE h_index = i * RESTART_TIMES + i; // todo: RESTART_TIMES == m
      // dot product using cg
      dot_prod_cg<INDEX_TYPE, DATA_TYPE, THREADS, ELES_PER_THREAD, STRIDES>(grid, gid, reg_x, reg_z, d_temp_buffer,
                                                                            &(d_h[h_index]));
      grid.sync();

      const DATA_TYPE alpha = -d_h[h_index];

      // register level daxpy
      register_daxpy<INDEX_TYPE, DATA_TYPE, ELES_PER_THREAD, STRIDES>(reg_x, reg_z, alpha);
    }
    grid.sync();

    const INDEX_TYPE h_idx = (i + 1) * RESTART_TIMES + i;
    calculate_norm_cg<INDEX_TYPE, DATA_TYPE, THREADS, ELES_PER_THREAD, STRIDES>(grid, gid, reg_x, d_temp_buffer,
                                                                                &(d_h[h_idx]));
    grid.sync();
    const DATA_TYPE alpha = 1 / d_h[h_idx];

    register_dscal<INDEX_TYPE, DATA_TYPE, ELES_PER_THREAD, STRIDES>(reg_x, alpha);

    // store back to array d_v[i+1]
    // d_r0: The d_r0 array is not used after the current loop, just need to store d_v[i+1].
    store_v_to_device<INDEX_TYPE, DATA_TYPE, ELES_PER_THREAD, STRIDES>(g_threads_num, gid, reg_x, d_v + (i + 1) * N, N);
  }
} // namespace usparse

template <typename INDEX_TYPE, typename DATA_TYPE, int THREADS, int ELES_PER_THREAD, int STRIDES, int OPT_LEVEL,
          int ITERS>
void usparse::gs_fusion_internal_imp(DATA_TYPE *d_r0, const DATA_TYPE *d_v, DATA_TYPE *d_h, DATA_TYPE *d_temp_buffer,
                                     const INDEX_TYPE N, const INDEX_TYPE m, const INDEX_TYPE i,
                                     std::size_t *temp_buffer_size_byte) {
  constexpr int ELES_PER_BLOCK = THREADS * ELES_PER_THREAD * STRIDES;
  INDEX_TYPE BLOCKS = N / ELES_PER_BLOCK + (N % ELES_PER_BLOCK == 0 ? 0 : 1);

  if (d_temp_buffer == nullptr) {
    *temp_buffer_size_byte = (i + 1) * (BLOCKS) * sizeof(DATA_TYPE);
    return;
  } else {
    void *kernelArgs[] = {(void *)&d_r0, (void *)&d_v, (void *)&d_h, (void *)&d_temp_buffer,
                          (void *)&N,    (void *)&m,   (void *)&i};
    hipError_t launchErr = hipLaunchCooperativeKernel(
        gs_fusion_kernel<INDEX_TYPE, DATA_TYPE, THREADS, ELES_PER_THREAD, STRIDES, OPT_LEVEL, ITERS>, BLOCKS, THREADS,
        kernelArgs, 0, 0);

    if (launchErr != hipSuccess) {
      std::cerr << "Kernel launch failed: " << hipGetErrorString(launchErr) << std::endl;
    }
  }
}

template <typename INDEX_TYPE, typename DATA_TYPE, int OPT_LEVEL>
void usparse::gs_fusion(DATA_TYPE *d_r0, const DATA_TYPE *d_v, DATA_TYPE *d_h, DATA_TYPE *d_temp_buffer,
                        const INDEX_TYPE N, const INDEX_TYPE m, const INDEX_TYPE i,
                        std::size_t *temp_buffer_size_byte) {

#define GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, ITERS)                                                 \
  gs_fusion_internal_imp<INDEX_TYPE, DATA_TYPE, THREADS, ELES_PER_THREAD, STRIDES, OPT_LEVEL, ITERS>(                  \
      d_r0, d_v, d_h, d_temp_buffer, N, m, i, temp_buffer_size_byte);
  if (N < 163840) { // 163840
    constexpr int THREADS = 512;
    constexpr int ELES_PER_THREAD = 4;
    constexpr int STRIDES = 2;
    if (i == 0) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 0);
    } else if (i == 1) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 1);
    } else if (i == 2) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 2);
    } else if (i == 3) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 3);
    } else if (i == 4) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 4);
    } else if (i == 5) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 5);
    } else if (i == 6) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 6);
    } else if (i == 7) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 7);
    } else if (i == 8) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 8);
    } else if (i == 9) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 9);
    } else if (i == 10) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 10);
    } else if (i == 11) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 11);
    } else if (i == 12) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 12);
    } else if (i == 13) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 13);
    } else if (i == 14) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 14);
    } else if (i == 15) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 15);
    } else if (i == 16) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 16);
    } else if (i == 17) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 17);
    } else if (i == 18) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 18);
    } else if (i == 19) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 19);
    } else if (i == 20) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 20);
    }
  } else if (N < 1000000) {
    constexpr int THREADS = 512;
    constexpr int ELES_PER_THREAD = 4;
    constexpr int STRIDES = 4;
    if (i == 0) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 0);
    } else if (i == 1) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 1);
    } else if (i == 2) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 2);
    } else if (i == 3) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 3);
    } else if (i == 4) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 4);
    } else if (i == 5) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 5);
    } else if (i == 6) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 6);
    } else if (i == 7) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 7);
    } else if (i == 8) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 8);
    } else if (i == 9) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 9);
    } else if (i == 10) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 10);
    } else if (i == 11) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 11);
    } else if (i == 12) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 12);
    } else if (i == 13) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 13);
    } else if (i == 14) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 14);
    } else if (i == 15) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 15);
    } else if (i == 16) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 16);
    } else if (i == 17) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 17);
    } else if (i == 18) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 18);
    } else if (i == 19) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 19);
    } else if (i == 20) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 20);
    }
  } else {
    constexpr int THREADS = 512;
    constexpr int ELES_PER_THREAD = 4;
    constexpr int STRIDES = 4;
    if (i == 0) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 0);
    } else if (i == 1) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 1);
    } else if (i == 2) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 2);
    } else if (i == 3) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 3);
    } else if (i == 4) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 4);
    } else if (i == 5) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 5);
    } else if (i == 6) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 6);
    } else if (i == 7) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 7);
    } else if (i == 8) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 8);
    } else if (i == 9) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 9);
    } else if (i == 10) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 10);
    } else if (i == 11) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 11);
    } else if (i == 12) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 12);
    } else if (i == 13) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 13);
    } else if (i == 14) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 14);
    } else if (i == 15) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 15);
    } else if (i == 16) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 16);
    } else if (i == 17) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 17);
    } else if (i == 18) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 18);
    } else if (i == 19) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 19);
    } else if (i == 20) {
      GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, 20);
    }
  }
#undef GS_FUSION_SHORT_MARK
}

#define GS_FUSION_SHORT_MARK(THREADS, ELES_PER_THREAD, STRIDES, ITERS)                                                 \
  template void usparse::gs_fusion_internal_imp<uint, float, THREADS, ELES_PER_THREAD, STRIDES,                        \
                                                usparse::GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION_LEVEL_2, ITERS>(       \
      float *, const float *, float *, float *, const uint, const uint, const uint, std::size_t *);
GS_FUSION_SHORT_MARK(128, 2, 2, 0);
GS_FUSION_SHORT_MARK(128, 2, 4, 0);
GS_FUSION_SHORT_MARK(128, 4, 2, 0);
GS_FUSION_SHORT_MARK(128, 4, 4, 0);

GS_FUSION_SHORT_MARK(256, 2, 2, 0);
GS_FUSION_SHORT_MARK(256, 2, 4, 0);
GS_FUSION_SHORT_MARK(256, 4, 2, 0);
GS_FUSION_SHORT_MARK(256, 4, 4, 0);

GS_FUSION_SHORT_MARK(512, 2, 2, 0);
GS_FUSION_SHORT_MARK(512, 2, 4, 0);
GS_FUSION_SHORT_MARK(512, 4, 2, 0);
GS_FUSION_SHORT_MARK(512, 4, 4, 0);

GS_FUSION_SHORT_MARK(128, 2, 2, 1);
GS_FUSION_SHORT_MARK(128, 2, 4, 1);
GS_FUSION_SHORT_MARK(128, 4, 2, 1);
GS_FUSION_SHORT_MARK(128, 4, 4, 1);

GS_FUSION_SHORT_MARK(256, 2, 2, 1);
GS_FUSION_SHORT_MARK(256, 2, 4, 1);
GS_FUSION_SHORT_MARK(256, 4, 2, 1);
GS_FUSION_SHORT_MARK(256, 4, 4, 1);

GS_FUSION_SHORT_MARK(512, 2, 2, 1);
GS_FUSION_SHORT_MARK(512, 2, 4, 1);
GS_FUSION_SHORT_MARK(512, 4, 2, 1);
GS_FUSION_SHORT_MARK(512, 4, 4, 1);

GS_FUSION_SHORT_MARK(128, 2, 2, 2);
GS_FUSION_SHORT_MARK(128, 2, 4, 2);
GS_FUSION_SHORT_MARK(128, 4, 2, 2);
GS_FUSION_SHORT_MARK(128, 4, 4, 2);

GS_FUSION_SHORT_MARK(256, 2, 2, 2);
GS_FUSION_SHORT_MARK(256, 2, 4, 2);
GS_FUSION_SHORT_MARK(256, 4, 2, 2);
GS_FUSION_SHORT_MARK(256, 4, 4, 2);

GS_FUSION_SHORT_MARK(512, 2, 2, 2);
GS_FUSION_SHORT_MARK(512, 2, 4, 2);
GS_FUSION_SHORT_MARK(512, 4, 2, 2);
GS_FUSION_SHORT_MARK(512, 4, 4, 2);

GS_FUSION_SHORT_MARK(128, 2, 2, 3);
GS_FUSION_SHORT_MARK(128, 2, 4, 3);
GS_FUSION_SHORT_MARK(128, 4, 2, 3);
GS_FUSION_SHORT_MARK(128, 4, 4, 3);

GS_FUSION_SHORT_MARK(256, 2, 2, 3);
GS_FUSION_SHORT_MARK(256, 2, 4, 3);
GS_FUSION_SHORT_MARK(256, 4, 2, 3);
GS_FUSION_SHORT_MARK(256, 4, 4, 3);

GS_FUSION_SHORT_MARK(512, 2, 2, 3);
GS_FUSION_SHORT_MARK(512, 2, 4, 3);
GS_FUSION_SHORT_MARK(512, 4, 2, 3);
GS_FUSION_SHORT_MARK(512, 4, 4, 3);

GS_FUSION_SHORT_MARK(128, 2, 2, 4);
GS_FUSION_SHORT_MARK(128, 2, 4, 4);
GS_FUSION_SHORT_MARK(128, 4, 2, 4);
GS_FUSION_SHORT_MARK(128, 4, 4, 4);

GS_FUSION_SHORT_MARK(256, 2, 2, 4);
GS_FUSION_SHORT_MARK(256, 2, 4, 4);
GS_FUSION_SHORT_MARK(256, 4, 2, 4);
GS_FUSION_SHORT_MARK(256, 4, 4, 4);

GS_FUSION_SHORT_MARK(512, 2, 2, 4);
GS_FUSION_SHORT_MARK(512, 2, 4, 4);
GS_FUSION_SHORT_MARK(512, 4, 2, 4);
GS_FUSION_SHORT_MARK(512, 4, 4, 4);

GS_FUSION_SHORT_MARK(128, 2, 2, 5);
GS_FUSION_SHORT_MARK(128, 2, 4, 5);
GS_FUSION_SHORT_MARK(128, 4, 2, 5);
GS_FUSION_SHORT_MARK(128, 4, 4, 5);

GS_FUSION_SHORT_MARK(256, 2, 2, 5);
GS_FUSION_SHORT_MARK(256, 2, 4, 5);
GS_FUSION_SHORT_MARK(256, 4, 2, 5);
GS_FUSION_SHORT_MARK(256, 4, 4, 5);

GS_FUSION_SHORT_MARK(512, 2, 2, 5);
GS_FUSION_SHORT_MARK(512, 2, 4, 5);
GS_FUSION_SHORT_MARK(512, 4, 2, 5);
GS_FUSION_SHORT_MARK(512, 4, 4, 5);

GS_FUSION_SHORT_MARK(128, 2, 2, 6);
GS_FUSION_SHORT_MARK(128, 2, 4, 6);
GS_FUSION_SHORT_MARK(128, 4, 2, 6);
GS_FUSION_SHORT_MARK(128, 4, 4, 6);

GS_FUSION_SHORT_MARK(256, 2, 2, 6);
GS_FUSION_SHORT_MARK(256, 2, 4, 6);
GS_FUSION_SHORT_MARK(256, 4, 2, 6);
GS_FUSION_SHORT_MARK(256, 4, 4, 6);

GS_FUSION_SHORT_MARK(512, 2, 2, 6);
GS_FUSION_SHORT_MARK(512, 2, 4, 6);
GS_FUSION_SHORT_MARK(512, 4, 2, 6);
GS_FUSION_SHORT_MARK(512, 4, 4, 6);

GS_FUSION_SHORT_MARK(128, 2, 2, 7);
GS_FUSION_SHORT_MARK(128, 2, 4, 7);
GS_FUSION_SHORT_MARK(128, 4, 2, 7);
GS_FUSION_SHORT_MARK(128, 4, 4, 7);

GS_FUSION_SHORT_MARK(256, 2, 2, 7);
GS_FUSION_SHORT_MARK(256, 2, 4, 7);
GS_FUSION_SHORT_MARK(256, 4, 2, 7);
GS_FUSION_SHORT_MARK(256, 4, 4, 7);

GS_FUSION_SHORT_MARK(512, 2, 2, 7);
GS_FUSION_SHORT_MARK(512, 2, 4, 7);
GS_FUSION_SHORT_MARK(512, 4, 2, 7);
GS_FUSION_SHORT_MARK(512, 4, 4, 7);

GS_FUSION_SHORT_MARK(128, 2, 2, 8);
GS_FUSION_SHORT_MARK(128, 2, 4, 8);
GS_FUSION_SHORT_MARK(128, 4, 2, 8);
GS_FUSION_SHORT_MARK(128, 4, 4, 8);

GS_FUSION_SHORT_MARK(256, 2, 2, 8);
GS_FUSION_SHORT_MARK(256, 2, 4, 8);
GS_FUSION_SHORT_MARK(256, 4, 2, 8);
GS_FUSION_SHORT_MARK(256, 4, 4, 8);

GS_FUSION_SHORT_MARK(512, 2, 2, 8);
GS_FUSION_SHORT_MARK(512, 2, 4, 8);
GS_FUSION_SHORT_MARK(512, 4, 2, 8);
GS_FUSION_SHORT_MARK(512, 4, 4, 8);

GS_FUSION_SHORT_MARK(128, 2, 2, 9);
GS_FUSION_SHORT_MARK(128, 2, 4, 9);
GS_FUSION_SHORT_MARK(128, 4, 2, 9);
GS_FUSION_SHORT_MARK(128, 4, 4, 9);

GS_FUSION_SHORT_MARK(256, 2, 2, 9);
GS_FUSION_SHORT_MARK(256, 2, 4, 9);
GS_FUSION_SHORT_MARK(256, 4, 2, 9);
GS_FUSION_SHORT_MARK(256, 4, 4, 9);

GS_FUSION_SHORT_MARK(512, 2, 2, 9);
GS_FUSION_SHORT_MARK(512, 2, 4, 9);
GS_FUSION_SHORT_MARK(512, 4, 2, 9);
GS_FUSION_SHORT_MARK(512, 4, 4, 9);

GS_FUSION_SHORT_MARK(128, 2, 2, 10);
GS_FUSION_SHORT_MARK(128, 2, 4, 10);
GS_FUSION_SHORT_MARK(128, 4, 2, 10);
GS_FUSION_SHORT_MARK(128, 4, 4, 10);

GS_FUSION_SHORT_MARK(256, 2, 2, 10);
GS_FUSION_SHORT_MARK(256, 2, 4, 10);
GS_FUSION_SHORT_MARK(256, 4, 2, 10);
GS_FUSION_SHORT_MARK(256, 4, 4, 10);

GS_FUSION_SHORT_MARK(512, 2, 2, 10);
GS_FUSION_SHORT_MARK(512, 2, 4, 10);
GS_FUSION_SHORT_MARK(512, 4, 2, 10);
GS_FUSION_SHORT_MARK(512, 4, 4, 10);

GS_FUSION_SHORT_MARK(128, 2, 2, 11);
GS_FUSION_SHORT_MARK(128, 2, 4, 11);
GS_FUSION_SHORT_MARK(128, 4, 2, 11);
GS_FUSION_SHORT_MARK(128, 4, 4, 11);

GS_FUSION_SHORT_MARK(256, 2, 2, 11);
GS_FUSION_SHORT_MARK(256, 2, 4, 11);
GS_FUSION_SHORT_MARK(256, 4, 2, 11);
GS_FUSION_SHORT_MARK(256, 4, 4, 11);

GS_FUSION_SHORT_MARK(512, 2, 2, 11);
GS_FUSION_SHORT_MARK(512, 2, 4, 11);
GS_FUSION_SHORT_MARK(512, 4, 2, 11);
GS_FUSION_SHORT_MARK(512, 4, 4, 11);

GS_FUSION_SHORT_MARK(128, 2, 2, 12);
GS_FUSION_SHORT_MARK(128, 2, 4, 12);
GS_FUSION_SHORT_MARK(128, 4, 2, 12);
GS_FUSION_SHORT_MARK(128, 4, 4, 12);

GS_FUSION_SHORT_MARK(256, 2, 2, 12);
GS_FUSION_SHORT_MARK(256, 2, 4, 12);
GS_FUSION_SHORT_MARK(256, 4, 2, 12);
GS_FUSION_SHORT_MARK(256, 4, 4, 12);

GS_FUSION_SHORT_MARK(512, 2, 2, 12);
GS_FUSION_SHORT_MARK(512, 2, 4, 12);
GS_FUSION_SHORT_MARK(512, 4, 2, 12);
GS_FUSION_SHORT_MARK(512, 4, 4, 12);

GS_FUSION_SHORT_MARK(128, 2, 2, 13);
GS_FUSION_SHORT_MARK(128, 2, 4, 13);
GS_FUSION_SHORT_MARK(128, 4, 2, 13);
GS_FUSION_SHORT_MARK(128, 4, 4, 13);

GS_FUSION_SHORT_MARK(256, 2, 2, 13);
GS_FUSION_SHORT_MARK(256, 2, 4, 13);
GS_FUSION_SHORT_MARK(256, 4, 2, 13);
GS_FUSION_SHORT_MARK(256, 4, 4, 13);

GS_FUSION_SHORT_MARK(512, 2, 2, 13);
GS_FUSION_SHORT_MARK(512, 2, 4, 13);
GS_FUSION_SHORT_MARK(512, 4, 2, 13);
GS_FUSION_SHORT_MARK(512, 4, 4, 13);

GS_FUSION_SHORT_MARK(128, 2, 2, 14);
GS_FUSION_SHORT_MARK(128, 2, 4, 14);
GS_FUSION_SHORT_MARK(128, 4, 2, 14);
GS_FUSION_SHORT_MARK(128, 4, 4, 14);

GS_FUSION_SHORT_MARK(256, 2, 2, 14);
GS_FUSION_SHORT_MARK(256, 2, 4, 14);
GS_FUSION_SHORT_MARK(256, 4, 2, 14);
GS_FUSION_SHORT_MARK(256, 4, 4, 14);

GS_FUSION_SHORT_MARK(512, 2, 2, 14);
GS_FUSION_SHORT_MARK(512, 2, 4, 14);
GS_FUSION_SHORT_MARK(512, 4, 2, 14);
GS_FUSION_SHORT_MARK(512, 4, 4, 14);

GS_FUSION_SHORT_MARK(128, 2, 2, 15);
GS_FUSION_SHORT_MARK(128, 2, 4, 15);
GS_FUSION_SHORT_MARK(128, 4, 2, 15);
GS_FUSION_SHORT_MARK(128, 4, 4, 15);

GS_FUSION_SHORT_MARK(256, 2, 2, 15);
GS_FUSION_SHORT_MARK(256, 2, 4, 15);
GS_FUSION_SHORT_MARK(256, 4, 2, 15);
GS_FUSION_SHORT_MARK(256, 4, 4, 15);

GS_FUSION_SHORT_MARK(512, 2, 2, 15);
GS_FUSION_SHORT_MARK(512, 2, 4, 15);
GS_FUSION_SHORT_MARK(512, 4, 2, 15);
GS_FUSION_SHORT_MARK(512, 4, 4, 15);

GS_FUSION_SHORT_MARK(128, 2, 2, 16);
GS_FUSION_SHORT_MARK(128, 2, 4, 16);
GS_FUSION_SHORT_MARK(128, 4, 2, 16);
GS_FUSION_SHORT_MARK(128, 4, 4, 16);

GS_FUSION_SHORT_MARK(256, 2, 2, 16);
GS_FUSION_SHORT_MARK(256, 2, 4, 16);
GS_FUSION_SHORT_MARK(256, 4, 2, 16);
GS_FUSION_SHORT_MARK(256, 4, 4, 16);

GS_FUSION_SHORT_MARK(512, 2, 2, 16);
GS_FUSION_SHORT_MARK(512, 2, 4, 16);
GS_FUSION_SHORT_MARK(512, 4, 2, 16);
GS_FUSION_SHORT_MARK(512, 4, 4, 16);

GS_FUSION_SHORT_MARK(128, 2, 2, 17);
GS_FUSION_SHORT_MARK(128, 2, 4, 17);
GS_FUSION_SHORT_MARK(128, 4, 2, 17);
GS_FUSION_SHORT_MARK(128, 4, 4, 17);

GS_FUSION_SHORT_MARK(256, 2, 2, 17);
GS_FUSION_SHORT_MARK(256, 2, 4, 17);
GS_FUSION_SHORT_MARK(256, 4, 2, 17);
GS_FUSION_SHORT_MARK(256, 4, 4, 17);

GS_FUSION_SHORT_MARK(512, 2, 2, 17);
GS_FUSION_SHORT_MARK(512, 2, 4, 17);
GS_FUSION_SHORT_MARK(512, 4, 2, 17);
GS_FUSION_SHORT_MARK(512, 4, 4, 17);

GS_FUSION_SHORT_MARK(128, 2, 2, 18);
GS_FUSION_SHORT_MARK(128, 2, 4, 18);
GS_FUSION_SHORT_MARK(128, 4, 2, 18);
GS_FUSION_SHORT_MARK(128, 4, 4, 18);

GS_FUSION_SHORT_MARK(256, 2, 2, 18);
GS_FUSION_SHORT_MARK(256, 2, 4, 18);
GS_FUSION_SHORT_MARK(256, 4, 2, 18);
GS_FUSION_SHORT_MARK(256, 4, 4, 18);

GS_FUSION_SHORT_MARK(512, 2, 2, 18);
GS_FUSION_SHORT_MARK(512, 2, 4, 18);
GS_FUSION_SHORT_MARK(512, 4, 2, 18);
GS_FUSION_SHORT_MARK(512, 4, 4, 18);

GS_FUSION_SHORT_MARK(128, 2, 2, 19);
GS_FUSION_SHORT_MARK(128, 2, 4, 19);
GS_FUSION_SHORT_MARK(128, 4, 2, 19);
GS_FUSION_SHORT_MARK(128, 4, 4, 19);

GS_FUSION_SHORT_MARK(256, 2, 2, 19);
GS_FUSION_SHORT_MARK(256, 2, 4, 19);
GS_FUSION_SHORT_MARK(256, 4, 2, 19);
GS_FUSION_SHORT_MARK(256, 4, 4, 19);

GS_FUSION_SHORT_MARK(512, 2, 2, 19);
GS_FUSION_SHORT_MARK(512, 2, 4, 19);
GS_FUSION_SHORT_MARK(512, 4, 2, 19);
GS_FUSION_SHORT_MARK(512, 4, 4, 19);

GS_FUSION_SHORT_MARK(128, 2, 2, 20);
GS_FUSION_SHORT_MARK(128, 2, 4, 20);
GS_FUSION_SHORT_MARK(128, 4, 2, 20);
GS_FUSION_SHORT_MARK(128, 4, 4, 20);

GS_FUSION_SHORT_MARK(256, 2, 2, 20);
GS_FUSION_SHORT_MARK(256, 2, 4, 20);
GS_FUSION_SHORT_MARK(256, 4, 2, 20);
GS_FUSION_SHORT_MARK(256, 4, 4, 20);

GS_FUSION_SHORT_MARK(512, 2, 2, 20);
GS_FUSION_SHORT_MARK(512, 2, 4, 20);
GS_FUSION_SHORT_MARK(512, 4, 2, 20);
GS_FUSION_SHORT_MARK(512, 4, 4, 20);
#undef GS_FUSION_SHORT_MARK

template void usparse::gs_fusion<uint, double, usparse::GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION_LEVEL_1>(
    double *d_r0, const double *d_v, double *d_h, double *d_temp_buffer, const uint N, const uint m, const uint i,
    std::size_t *temp_buffer_size_byte);

template void usparse::gs_fusion<uint, float, usparse::GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION_LEVEL_1>(
    float *d_r0, const float *d_v, float *d_h, float *d_temp_buffer, const uint N, const uint m, const uint i,
    std::size_t *temp_buffer_size_byte);

template void usparse::gs_fusion<uint, double, usparse::GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION_LEVEL_2>(
    double *d_r0, const double *d_v, double *d_h, double *d_temp_buffer, const uint N, const uint m, const uint i,
    std::size_t *temp_buffer_size_byte);

template void usparse::gs_fusion<uint, float, usparse::GMRES_OPT_USE_INNER_LOOP_KERNEL_FUSION_LEVEL_2>(
    float *d_r0, const float *d_v, float *d_h, float *d_temp_buffer, const uint N, const uint m, const uint i,
    std::size_t *temp_buffer_size_byte);
