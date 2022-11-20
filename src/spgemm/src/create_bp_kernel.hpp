#ifndef USTB_SPARSE_SPGEMM_CREATE_BP_KERNEL_HPP
#define USTB_SPARSE_SPGEMM_CREATE_BP_KERNEL_HPP

#include <hip/hip_runtime.h>

template <typename I, size_t CHUNK_SIZE>
__global__ void create_bp_wf_per_row_kernel(const size_t m, const I *row_ptr, const I *col_idx_ptr, I *bp,
                                            const size_t BP_IDX_PER_ROW) {
  // thread id in global
  const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  // thread num in global
  const int global_thread_num = gridDim.x * blockDim.x;
  // wf id in global
  const int global_wf_id = global_thread_id / __WF_SIZE__;
  // wf num in global
  const int global_wf_num = global_thread_num / __WF_SIZE__;
  // thread id in wf
  const int wf_thread_id = global_thread_id % __WF_SIZE__;
  // thread num in wf
  const int wf_thread_num = __WF_SIZE__;
  // each wf process a row
  for (int row = global_wf_id; row < m; row += global_wf_num) {
    I *row_bp_ptr = bp + BP_IDX_PER_ROW * row;
    I base_bp = CHUNK_SIZE;
    int bp_idx = 0;
    const I row_idx_begin = row_ptr[row];
    const I row_idx_end = row_ptr[row + 1];
    const int WF_ROUNDS = (row_idx_end - row_idx_begin + __WF_SIZE__ - 1) / __WF_SIZE__;
    for (int i = 0; i < WF_ROUNDS; i++) {
      const int idx = row_idx_begin + __WF_SIZE__ * i + wf_thread_id;
      const I col = idx < row_idx_end ? col_idx_ptr[idx] : static_cast<I>(0);
      bool flag = col >= base_bp;
      if (__any(flag)) {
        do {
          I tmp_idx = flag ? idx : row_idx_end;
          for (int i = __WF_SIZE__ >> 1; i > 0; i >>= 1) {
            tmp_idx = min(__shfl_down(tmp_idx, i), tmp_idx);
          }
          tmp_idx = __shfl(tmp_idx, 0);
          I tmp_col = col_idx_ptr[tmp_idx];
          const int bp_idx_begin = bp_idx;
          const int bp_idx_end = tmp_col / CHUNK_SIZE + 1;
          for (int i = bp_idx_begin + wf_thread_id; i < bp_idx_end; i += wf_thread_num) {
            row_bp_ptr[i] = tmp_idx;
          }
          bp_idx = bp_idx_end;
          base_bp = bp_idx_end * CHUNK_SIZE;
          flag = col >= base_bp;
        } while (__any(flag));
      }
    }
    if (wf_thread_id == 0) {
      row_bp_ptr[0] = row_idx_begin;
    }
    bp_idx = __shfl(bp_idx, 0);
    for (int idx = bp_idx + wf_thread_id; idx < BP_IDX_PER_ROW; idx += wf_thread_num) {
      row_bp_ptr[idx] = row_idx_end;
    }
  }
}

#endif // USTB_SPARSE_SPGEMM_CREATE_BP_KERNEL_HPP