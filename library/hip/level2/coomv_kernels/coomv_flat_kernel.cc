/**
 * Created by genshen on 2025/9/23.
 *
 */

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "../../level1/usparse_blas_level1.h"
#include "../../primitives/reduce/wavefront/wave_reduce_by_key.hpp"
#include "building_config.h"
#include "coomv_flat_kernel.h"

/**
 * @param m: total rows in the matrix.
 * @param nnz: total nnz in the matrix.
 */
template <int WF_SIZE, int R, int THREADS, typename I, typename T>
__global__ void __launch_bounds__(512)
    coomv_flat_one_pass_kernel(const I m, const I nnz, const I *__restrict__ coo_row_ind,
                               const I *__restrict__ coo_col_ind, const T *__restrict__ coo_val, const T alpha,
                               const T beta, const T *__restrict__ x, T *__restrict__ y) {
  const int global_thread_id = threadIdx.x + THREADS * blockIdx.x;
  const int global_threads_num = hipGridDim_x * THREADS;

  const int wf_id_in_block = blockIdx.x / WF_SIZE; // wavefront id in block
  const int block_id = blockIdx.x;                 // block id
  const int tid_in_block = threadIdx.x % THREADS;  // thread id in one block

  constexpr int NNZ_PER_WAVE = WF_SIZE * R;
  constexpr int NNZ_PER_BLOCK = THREADS * R;
  // __shared__ T shared_val[nnz_per_block];
  const I last_element_index = nnz;

  I bp_index = block_id;
  // each block process THREADS * R elements.
  const int block_base = NNZ_PER_BLOCK * block_id;

  T thread_reg_data[R];  // the R multip results in current thread.
  I thread_row_index[R]; // the R row indexes in current thread.
#pragma unroll
  for (int i = 0; i < R; i++) {
    const I shared_inx = tid_in_block + i * THREADS;
    const I index = block_base + shared_inx;
    if (index < last_element_index) { // TODO: remove this if-else can improve performance.
                                      // shared_val[shared_inx] = coo_val[index] * x[csr_col_ind[index]];
      thread_reg_data[i] = coo_val[index] * x[coo_col_ind[index]];
      thread_row_index[i] = coo_row_ind[index];
    } else {
      thread_reg_data[i] = static_cast<T>(0);
      thread_row_index[i] = static_cast<I>(0);
    }
  }

  // reduce in a wavefront and store results to device memory
  for (int i = 0; i < R; i++) {
    const T val = alpha * thread_reg_data[i];
    usparse::primitives::reduce::wavefront_reduce_by_key<I, T, I, WF_SIZE>(threadIdx.x % WF_SIZE, val,
                                                                           thread_row_index[i], y, 0);
  }
}

// TODO: remember to merge with csrmv
template <typename DATA_TYPE, typename INDEX_TYPE, int BEAT_Y_POLICY>
inline void coo_flat_set_beta_y(DATA_TYPE *y, const INDEX_TYPE m, const DATA_TYPE beta) {
  if (BEAT_Y_POLICY == BEAT_Y_POLICY_AUTO) {
    // set beta*y.
    if (beta == 0.0) {
      // set vector y to 0. because: y = A*x + 0.0 *y.
      usparse::device_scale_set<INDEX_TYPE, DATA_TYPE>(0.0, y, m);
    } else if (beta == 1.0) {
      // do nothing. because flat is based on atomicAdd.
    } else {
      usparse::device_dscal(beta, y, m);
    }
  } else if (BEAT_Y_POLICY == BEAT_Y_POLICY_ZERO) {
    usparse::device_scale_set<INDEX_TYPE, DATA_TYPE>(0.0, y, m);
  } else if (BEAT_Y_POLICY == BEAT_Y_POLICY_ONE) {
    return; // nothing to do.
  } else {  // BEAT_Y_POLICY == BEAT_Y_POLICY_OTHER
    usparse::device_dscal(beta, y, m);
  }
}

template <typename INDEX_TYPE, typename DATA_TYPE, int BEAT_Y_POLICY>
void adaptive_flat_coo_sparse_spmv(int trans, const DATA_TYPE alpha, const DATA_TYPE beta,
                                   const usparse::coo_desc<INDEX_TYPE, DATA_TYPE> d_coo_desc, const DATA_TYPE *x,
                                   DATA_TYPE *y) {
  constexpr int THREADS_PER_BLOCK = 512;
  constexpr int R = 2;
  const INDEX_TYPE nnz = d_coo_desc.nnz;

  coo_flat_set_beta_y<DATA_TYPE, INDEX_TYPE, BEAT_Y_POLICY>(y, d_coo_desc.rows, beta);

  const int HIP_BLOCKS = nnz / (R * THREADS_PER_BLOCK) + ((nnz % (R * THREADS_PER_BLOCK) == 0) ? 0 : 1);
  (coomv_flat_one_pass_kernel<__WF_SIZE__, R, THREADS_PER_BLOCK, INDEX_TYPE,
                              DATA_TYPE>)<<<HIP_BLOCKS, THREADS_PER_BLOCK>>>(d_coo_desc.rows, d_coo_desc.nnz,
                                                                             d_coo_desc.row_index, d_coo_desc.col_index,
                                                                             d_coo_desc.values, alpha, beta, x, y);
}

template void
adaptive_flat_coo_sparse_spmv<int, double, BEAT_Y_POLICY_AUTO>(int trans, const double alpha, const double beta,
                                                               const usparse::coo_desc<int, double> d_coo_desc,
                                                               const double *x, double *y);

template void
adaptive_flat_coo_sparse_spmv<int, float, BEAT_Y_POLICY_AUTO>(int trans, const float alpha, const float beta,
                                                              const usparse::coo_desc<int, float> d_coo_desc,
                                                              const float *x, float *y);
