//
// Created by shenjihao on 2025/07/21.
//

#include "coo2csr.h"

#include "building_config.h"
#include "include/usparse.h"

namespace usparse {
  namespace conversion {
    template <typename INDEX_TYPE, typename IN_TYPE, typename OUT_TYPE, int WAVE_SIZE>
    __global__ void coo2csr_kernel_part1(const INDEX_TYPE m, const INDEX_TYPE nnz, const IN_TYPE *coo_row_ind,
                                         OUT_TYPE *csr_row_ptr) {

      const int tid = blockIdx.x * blockDim.x + threadIdx.x;
      const int lane = tid % WAVE_SIZE;
      if (tid == 0) {
        csr_row_ptr[0] = 0;
        csr_row_ptr[m] = nnz;
      }
      if (tid >= nnz) {
        return;
      }
      int current_row = coo_row_ind[tid];
      int next_row = __shfl_down(current_row, 1);
      if (lane == WAVE_SIZE - 1 && tid != nnz - 1) {
        next_row = coo_row_ind[tid + 1];
      }
      if (tid == nnz - 1) {
        next_row = current_row;
      }
      if (current_row != next_row) {
        csr_row_ptr[next_row] = tid + 1;
      }
    }

    template <typename INDEX_TYPE, typename IN_TYPE, typename OUT_TYPE, int WAVE_SIZE>
    __global__ void coo2csr_kernel_part2(const INDEX_TYPE m, OUT_TYPE *csr_row_ptr) {
      const int tid = blockIdx.x * blockDim.x + threadIdx.x;
      if ((tid == 0) || (tid == 1) || (tid > m)) {
        return;
      }
      if ((csr_row_ptr[tid - 1] == 0) && (csr_row_ptr[tid] != 0)) {
        int flag = tid - 1;
        int value = csr_row_ptr[tid];
        while (flag > 0 && csr_row_ptr[flag] == 0) {
          csr_row_ptr[flag] = value;
          flag--;
        }
      }
    }
  } // namespace conversion
} // namespace usparse

template <typename INDEX_TYPE, typename IN_TYPE, typename OUT_TYPE>
void usparse::conversion::coo2csr(const INDEX_TYPE m, const INDEX_TYPE nnz, const IN_TYPE *coo_row_ind,
                                  OUT_TYPE *csr_row_ptr) {
  constexpr int THREAD_PER_BLOCK = 512;
  int blocks = (nnz - 1) / THREAD_PER_BLOCK + 1;
  (usparse::conversion::coo2csr_kernel_part1<INDEX_TYPE, IN_TYPE, OUT_TYPE, __WF_SIZE__>)<<<blocks, THREAD_PER_BLOCK>>>(
      m, nnz, coo_row_ind, csr_row_ptr);
  (usparse::conversion::coo2csr_kernel_part2<INDEX_TYPE, IN_TYPE, OUT_TYPE, __WF_SIZE__>)<<<blocks, THREAD_PER_BLOCK>>>(
      m, csr_row_ptr);
  return;
}

template void usparse::conversion::coo2csr<usparse::usparse_int, usparse::usparse_int, usparse::usparse_int>(
    const usparse::usparse_int m, const usparse::usparse_int nnz, const usparse::usparse_int *coo_row_ind,
    usparse::usparse_int *csr_row_ptr);

template void usparse::conversion::coo2csr<usparse::usparse_uint, usparse::usparse_uint, usparse::usparse_uint>(
    const usparse::usparse_uint m, const usparse::usparse_uint nnz, const usparse::usparse_uint *coo_row_ind,
    usparse::usparse_uint *csr_row_ptr);
