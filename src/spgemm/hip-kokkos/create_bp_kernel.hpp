#ifndef SPGEMM_KOKKOS_CREATE_BP_KERNEL_HPP
#define SPGEMM_KOKKOS_CREATE_BP_KERNEL_HPP

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
  // each vector process a row
  for (int row = global_wf_id; row < m; row += global_wf_num) {
    I *row_bp_ptr = &bp[BP_IDX_PER_ROW * row];
    I base_bp = CHUNK_SIZE;
    int bp_idx = 0;
    const I row_idx_begin = row_ptr[row];
    const I row_idx_end = row_ptr[row + 1];

    if (wf_thread_id == 0) {
      row_bp_ptr[bp_idx++] = row_idx_begin;
    }

    for (int idx = row_idx_begin + wf_thread_id; idx < row_idx_end; idx += wf_thread_num) {
      const I col = col_idx_ptr[idx];
      bool flag = col >= base_bp;
      if (__any(flag)) {
        do {
          I tmp_idx = flag ? idx : row_idx_end;
          for (int i = __WF_SIZE__ >> 1; i > 0; i >>= 1) {
            tmp_idx = min(__shfl_down(tmp_idx, i), tmp_idx);
          }
          if (wf_thread_id == 0) {
            row_bp_ptr[bp_idx++] = tmp_idx;
          }
          base_bp += CHUNK_SIZE;
          flag = col >= base_bp;
        } while (__any(flag));
      }
    }
    bp_idx = __shfl(bp_idx, 0);
    for (int idx = bp_idx + wf_thread_id; idx < BP_IDX_PER_ROW; idx += wf_thread_num) {
      row_bp_ptr[idx] = row_idx_end;
    }
  }
}

template <typename I, size_t CHUNK_SIZE, size_t BUFFER_SIZE>
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

  I local_bp[BUFFER_SIZE];
  // each vector process a row
  for (int row = global_wf_id; row < m; row += global_wf_num) {
    I *row_bp_ptr = &bp[BP_IDX_PER_ROW * row];
    I base_bp = CHUNK_SIZE;
    int bp_idx = 1;
    const I row_idx_begin = row_ptr[row];
    const I row_idx_end = row_ptr[row + 1];

    if (wf_thread_id == 0) {
      row_bp_ptr[0] = row_idx_begin;
    }
    // init local bp
    for (int i = 0; i < BUFFER_SIZE; i++) {
      local_bp[i] = row_idx_end;
    }
    for (int idx = row_idx_begin + wf_thread_id; idx < row_idx_end; idx += wf_thread_num) {
      const I col = col_idx_ptr[idx];
      while (col >= base_bp) {
        local_bp[bp_idx++] = idx;
        base_bp += CHUNK_SIZE;
      }
    }
    int max_bp_idx = bp_idx;
    for (int i = __WF_SIZE__ >> 1; i > 0; i >>= 1) {
      max_bp_idx = max(__shfl_down(max_bp_idx, i), max_bp_idx);
    }
    max_bp_idx = __shfl(max_bp_idx, 0);

    for (int i = 1; i < max_bp_idx; i++) {
      I bp_val = local_bp[i];
      for (int j = __WF_SIZE__ >> 1; j > 0; j >>= 1) {
        bp_val = min(__shfl_down(bp_val, j), bp_val);
      }
      if (wf_thread_id == 0) {
        row_bp_ptr[i] = bp_val;
      }
    }
    for (int idx = max_bp_idx + wf_thread_id; idx < BP_IDX_PER_ROW; idx += wf_thread_num) {
      row_bp_ptr[idx] = row_idx_end;
    }
  }
}

#endif // SPGEMM_KOKKOS_CREATE_BP_KERNEL_HPP