//
// Created by genshen on 2025/10/08.
//

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "building_config.h"
#include "include/usparse_matrix.h"
#include "include/usparse_types.h"

#ifdef __USPARSE_BUILD_BACKEND_ROCM__
#define __ballot_sync(mask, pred) __ballot(pred)
#endif

namespace usparse {
  namespace internal {
        using DENSE_ORDER = int;
    constexpr DENSE_ORDER DENSE_ORDER_COLUMN = 0;
    constexpr DENSE_ORDER DENSE_ORDER_ROW = 1;
    namespace csrmm_kernels {
      /**
       * This algirithm come from paper "Design Principles for Sparse Matrix Multiplication",
       * https://doi.org/10.1007/978-3-319-96983-1_48, Euro-Par 2018.
       *
       * The code implementation is inspired and modified from github repo "merge-spmm" (Apache-2.0 license):
       * https://github.com/owensgroup/merge-spmm/blob/da5c55ad254038c061b6386df83dc0c42cc98b54/graphblas/backend/apspie/kernels/spmm.hpp#L244
       * kernel function: spmmRowKernel2<<<>>>
       *
       * Similar implementation: csrmmnn_template_row_split in
       * https://github.com/ROCm/rocSPARSE/blob/57ab6fadeaa55fead8e69f9a67b489502f35ac17/library/src/level3/csrmm_device_row_split.h#L153
       * It uses shared memory or other way instead.
       * For rocsparse, it conntains 2 kernels: csrmmnn_row_split_device,  csrmmnn_row_split_shared_device
       */
      template <typename INDEX_TYPE, typename DATA_TYPE, DENSE_ORDER B_ORDER, DENSE_ORDER C_ORDER, int TB, int VEC_SIZE,
                bool TRANS, bool BALLOT_SUPPORTED>
      __global__ void csrmmnn_row_split_kernel(const DATA_TYPE alpha, const DATA_TYPE beta, const INDEX_TYPE A_nrows,
                                               const INDEX_TYPE A_ncols, const INDEX_TYPE A_nnz,
                                               const INDEX_TYPE B_ncols, const INDEX_TYPE *A_csr_row_ptr,
                                               const INDEX_TYPE *A_csr_col_ind, const DATA_TYPE *A_csr_val,
                                               const DATA_TYPE *B_denseVal, DATA_TYPE *C_denseVal) {
        DATA_TYPE vals[TB];
        INDEX_TYPE col_all[TB];
        DATA_TYPE val_all[TB];
        const INDEX_TYPE &BC_ncols = B_ncols;
        const INDEX_TYPE &B_rows = A_ncols;

        const int thread_id = blockDim.x * blockIdx.x + threadIdx.x; // global thread idx
        const int warp_id = thread_id / VEC_SIZE;                    // global warp idx
        const int lane_id = thread_id % VEC_SIZE;
        int row = warp_id; // each wave for a csr row

        // TODO: B offset can be : ((blockIdx.y & 1) * WF_SIZE);
        const int BC_col_id = (blockIdx.y * VEC_SIZE) + lane_id;
        INDEX_TYPE C_offset;
        if (C_ORDER == DENSE_ORDER_COLUMN) { // trans
          C_offset = BC_col_id * A_nrows + row;
        } else { // no trans
          // result to C[row][Block.y * VEC_SIZE + lane_id]
          C_offset = (row * BC_ncols) + BC_col_id;
        }

        if (BC_col_id >= BC_ncols) {
          return;
        }

        if (row < A_nrows) {
          const INDEX_TYPE row_start = __ldg(A_csr_row_ptr + row);
          const INDEX_TYPE row_end = __ldg(A_csr_row_ptr + row + 1);

          INDEX_TYPE col = -1;
          DATA_TYPE val = static_cast<DATA_TYPE>(0);
          DATA_TYPE sum = static_cast<DATA_TYPE>(0);
          INDEX_TYPE jj = row_start + lane_id;

          for (INDEX_TYPE jj_start = row_start; jj_start < row_end; jj_start += VEC_SIZE) {
            // load matrix A in a wavefront:
            int predicate = 0;
            if (jj < row_end) {
              predicate = 1;
              col = (__ldg(A_csr_col_ind + jj)); // * VEC_SIZE * 2; // << 6;
              val = __ldg(A_csr_val + jj);
            } else {
              col = static_cast<INDEX_TYPE>(0);
              val = static_cast<DATA_TYPE>(0);
            }
            jj += VEC_SIZE;

            // const unsigned long non_zero_bits = __ballot(VEC_SIZE, predicate);
            const unsigned long non_zero_bits =
                __ballot_sync(0xffffffff, predicate); // VECTOR_SIZE must be 64 here. Because it does not support mask.
            const unsigned int num_non_zeros_in_vector = __popcll(non_zero_bits);

            if (BALLOT_SUPPORTED) { // currently, it only supppr B_Row_Order and C_Row_Order
              const DATA_TYPE *B_offset = B_denseVal + BC_col_id;
              // load matrix B: a wavefront collaboratively load values in row using col index of A.
              for (int kk = 0; kk < VEC_SIZE; kk += TB) {
                // TB=2
                // lane:  0,     1,    2,   ...   ,  63   (one for 1 thread)
                // kk:    0,     2,    4,   ...   ,  62   (thread private)
                // ii:  [0,1], [0,1], [0,1], ..., [0,1]   (one for 1 thread)
                // ii+kk:[0,1], [2,3], [4,5], ..., [62,63] (thread private)
#pragma unroll
                for (int ii = 0; ii < TB; ii++) {
                  col_all[ii] = __shfl(col, ii + kk, VEC_SIZE); // bcast
                  val_all[ii] = __shfl(val, ii + kk, VEC_SIZE);
                  vals[ii] = val_all[ii] * __ldg(B_offset + B_ncols * col_all[ii]);
                }

#pragma unroll
                for (int ii = 0; ii < TB; ii++) {
                  sum += vals[ii];
                }
              }
            } else { // else: LOOP_ROLLING
#define ROOL(NN)                                                                                                       \
  for (int kk = 0; kk < NN; kk++) {                                                                                    \
    const INDEX_TYPE col_inx = __shfl(col, 0 + kk, VEC_SIZE);                                                          \
    const DATA_TYPE val_ = __shfl(val, 0 + kk, VEC_SIZE);                                                              \
    if (B_ORDER == DENSE_ORDER_COLUMN) {                                                                               \
      const DATA_TYPE *B_index = B_denseVal + B_rows * BC_col_id + col_inx;                                            \
      sum += val_ * __ldg(B_index);                                                                                    \
    } else {                                                                                                           \
      const DATA_TYPE *B_index = B_denseVal + BC_col_id + B_ncols * col_inx;                                           \
      sum += val_ * __ldg(B_index);                                                                                    \
    }                                                                                                                  \
  }

              // use loop unrolling
              switch (num_non_zeros_in_vector) {
              case 0:
                break;
              case 1:
                ROOL(1);
                break;
              case 2:
                ROOL(2);
                break;
              case 3:
                ROOL(3);
                break;
              case 4:
                ROOL(4);
                break;
              case 5:
                ROOL(5);
                break;
              case 6:
                ROOL(6);
                break;
              case 7:
                ROOL(7);
                break;
              case 8:
                ROOL(8);
                break;
              case 9:
                ROOL(9);
                break;
              case 10:
                ROOL(10);
                break;
              case 11:
                ROOL(11);
                break;
              case 12:
                ROOL(12);
                break;
              case 13:
                ROOL(13);
                break;
              case 14:
                ROOL(14);
                break;
              case 15:
                ROOL(15);
                break;
              case 16:
                ROOL(16);
                break;
              case 17:
                ROOL(17);
                break;
              case 18:
                ROOL(18);
                break;
              case 19:
                ROOL(19);
                break;
              case 20:
                ROOL(20);
                break;
              case 21:
                ROOL(21);
                break;
              case 22:
                ROOL(22);
                break;
              case 23:
                ROOL(23);
                break;
              case 24:
                ROOL(24);
                break;
              case 25:
                ROOL(25);
                break;
              case 26:
                ROOL(26);
                break;
              case 27:
                ROOL(27);
                break;
              case 28:
                ROOL(28);
                break;
              case 29:
                ROOL(29);
                break;
              case 30:
                ROOL(30);
                break;
              case 31:
                ROOL(31);
                break;
              case 32:
                ROOL(32);
                break;
              default:
                for (int kk = 0; kk < num_non_zeros_in_vector; kk++) {
                  const INDEX_TYPE col_inx = __shfl(col, 0 + kk, VEC_SIZE); // bcast
                  const DATA_TYPE val_ = __shfl(val, 0 + kk, VEC_SIZE);
                  if (B_ORDER == DENSE_ORDER_COLUMN) {
                    const DATA_TYPE *B_index = B_denseVal + B_rows * BC_col_id + col_inx;
                    sum += val_ * __ldg(B_index);
                  } else {
                    const DATA_TYPE *B_index = B_denseVal + BC_col_id + B_ncols * col_inx;
                    sum += val_ * __ldg(B_index);
                  }
                }
                break;
              }
#undef ROOL
            }
          }
          C_denseVal[C_offset] = alpha * sum; // TODO: + beta * C_denseVal[C_offset];
        }
        return;
      }
    } // namespace csrmm_kernels

    template <typename INDEX_TYPE, typename DATA_TYPE, int WF_SIZE>
    void adaptive_csrmm_vetor_row(const usparse::usparse_operation A_trans, const usparse::usparse_operation B_trans,
                                  const DATA_TYPE alpha, const DATA_TYPE beta,
                                  const usparse::csr_desc<INDEX_TYPE, DATA_TYPE> A_csr, const INDEX_TYPE B_cols,
                                  const DATA_TYPE *B_dense_val, DATA_TYPE *C_dense_val) {
      constexpr int TB = 2;
      constexpr int VECTOR_SIZE = WF_SIZE;
      constexpr int THREADS = 128;
      constexpr int ROWS_PER_BLOCK = THREADS / VECTOR_SIZE;

      const int BLOCKS_X = A_csr.rows / ROWS_PER_BLOCK + (A_csr.rows % ROWS_PER_BLOCK == 0 ? 0 : 1);
      const int BLOCKS_Y = B_cols / VECTOR_SIZE + (B_cols % VECTOR_SIZE == 0 ? 0 : 1);
      csrmm_kernels::csrmmnn_row_split_kernel<INDEX_TYPE, DATA_TYPE, DENSE_ORDER_ROW, DENSE_ORDER_ROW, TB,
                                              VECTOR_SIZE, false, false>
          <<<dim3(BLOCKS_X, BLOCKS_Y), THREADS>>>(alpha, beta, A_csr.rows, A_csr.cols, A_csr.nnz, B_cols, A_csr.row_ptr,
                                                  A_csr.col_index, A_csr.values, B_dense_val, C_dense_val);
      return;
    }
  } // namespace internal
} // namespace usparse

template void usparse::internal::adaptive_csrmm_vetor_row<int, double, __WF_SIZE__>(
    const usparse::usparse_operation A_trans, const usparse::usparse_operation B_trans, const double alpha,
    const double beta, const usparse::csr_desc<int, double> A_csr, const int B_cols, const double *B_dense_val,
    double *C_dense_val);

template void usparse::internal::adaptive_csrmm_vetor_row<int, float, __WF_SIZE__>(
    const usparse::usparse_operation A_trans, const usparse::usparse_operation B_trans, const float alpha,
    const float beta, const usparse::csr_desc<int, float> A_csr, const int B_cols, const float *B_dense_val,
    float *C_dense_val);
