#include "csrsm_kernels.h"

template <typename T>
__global__ void sptrsm_updown_kernel(const int *__restrict__ row_ptr, const int *__restrict__ col_idx,
                                     const T *__restrict__ val, const T *__restrict__ b, T *x, int *is_solved, int n,
                                     int rhs, bool is_lower) {
  // 计算原始warp索引
  int original_wrp = (hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x) / WARP_SIZE;
  // 根据三角类型调整处理顺序
  int wrp = is_lower ? original_wrp : (n - 1 - original_wrp);
  if (wrp >= n || wrp < 0)
    return;
  int lane = hipThreadIdx_x & (WARP_SIZE - 1); // 线程在warp内的编号
  int row_start = row_ptr[wrp];
  int row_end = (wrp == n - 1) ? row_ptr[n] : row_ptr[wrp + 1];
  // 确定对角线位置
  int diag_offset = is_lower ? (row_end - 1) : row_start;
  T piv = 1.0 / val[diag_offset]; // 对角线元素倒数

  // 根据三角类型确定计算范围
  int start_offset, end_offset;
  if (is_lower) {
    start_offset = row_start;
    end_offset = diag_offset; // 下三角处理到对角线前
  } else {
    start_offset = diag_offset + 1; // 上三角从对角线后开始
    end_offset = row_end;
  }

  for (int i = 0; i < rhs; ++i) {
    T sum = (lane == 0) ? b[wrp * rhs + i] : 0.0;

    // 处理非对角线元素（修正循环范围）
    for (int off = start_offset + lane; off < end_offset; off += WARP_SIZE) {
      if (off >= row_end)
        break; // 防止越界

      int col = col_idx[off];
      bool valid_dep = is_lower ? (col < wrp) : (col > wrp);

      if (valid_dep) {
        // 等待依赖行完成
        while (__atomic_load_n(&is_solved[col * rhs + i], __ATOMIC_ACQUIRE) == 0) {
          __builtin_amdgcn_s_sleep(1);
        }
        sum -= val[off] * x[col * rhs + i];
      }
    }

    // Warp内归约求和
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
      sum += __shfl_xor(sum, mask);
    }

    // 写入结果并标记完成
    if (lane == 0) {
      x[wrp * rhs + i] = sum * piv;
      __threadfence_block();
      __atomic_store_n(&is_solved[wrp * rhs + i], 1, __ATOMIC_RELEASE);
    }
  }
}

// 显式实例化
template __global__ void sptrsm_updown_kernel<float>(const int *, const int *, const float *, const float *, float *,
                                                     int *, int, int, bool);
template __global__ void sptrsm_updown_kernel<double>(const int *, const int *, const double *, const double *,
                                                      double *, int *, int, int, bool);
