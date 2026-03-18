//
// Created by genshen on 2021/7/15.
//

#include <cstdint>
#include <iostream>

#include "../../../level1/usparse_blas_level1.h"
#include "common/macros.h"
#include "common/usparse_utils.hpp"
#include "building_config.h"
#include "flat_config.h"
#include "flat_kernels_imp_multi_pass.hpp"
#include "flat_kernels_imp_one_pass.hpp"
#include "flat_kernels_imp_pre_calc.hpp"
#include "csrmv_flat.h"

/**
 * note: multi pass does not support FLAT_STORE_Y_POLICY_SEGMENTED_SUM mode currently.
 */
template <typename HIP_BLOCK_NUM_TYPE, typename SHARE_MEM_INDEX_TYPE, typename BP_INDEX_TYPE, typename BP_DATA_TYPE,
          typename A_TYPE, typename X_TYPE, typename Y_TYPE, typename COMPUTE_MULTI_TYPE, typename COMPUTE_SUM_TYPE,
          typename SCALE_DATA_TYPE, typename NNZ_TYPE, typename ROW_TYPE, typename COLUMN_TYPE, int R,
          int REDUCE_OPTION, int REDUCE_VEC_SIZE, int BLOCKS, int THREADS_PER_BLOCK>
inline void flat_multi_pass_sparse_spmv(hipStream_t stream, int trans, const SCALE_DATA_TYPE alpha,
                                        const SCALE_DATA_TYPE beta, ROW_TYPE m, COLUMN_TYPE n, NNZ_TYPE nnz,
                                        const NNZ_TYPE *rowptr, const COLUMN_TYPE *colindex, const A_TYPE *value,
                                        const X_TYPE *x, Y_TYPE *y) {
  BP_DATA_TYPE *break_points;
  // the nnz is rowptr[m], in one round, it can process about `blocks * R * threads_per_block` nnz.
  const NNZ_TYPE total_rounds =
      nnz / (R * THREADS_PER_BLOCK * BLOCKS) + (nnz % (R * THREADS_PER_BLOCK * BLOCKS) == 0 ? 0 : 1);
  // each round and each block both have a break point.
  const BP_INDEX_TYPE break_points_len = total_rounds * BLOCKS + 1;
  USPARSE_HIP_API_CHECK(hipMallocAsync((void **)&break_points, break_points_len * sizeof(BP_DATA_TYPE), stream));
  USPARSE_HIP_API_CHECK(hipMemsetAsync(break_points, 0, break_points_len * sizeof(BP_DATA_TYPE), stream));

  (pre_calc_break_point<ROW_TYPE, NNZ_TYPE, BP_DATA_TYPE, R * THREADS_PER_BLOCK>)<<<1024, 512, 0, stream>>>(
      rowptr, m, break_points, break_points_len);

  (csrmv_flat_multi_pass_kernel<__WF_SIZE__, R, REDUCE_OPTION, REDUCE_VEC_SIZE, BLOCKS, THREADS_PER_BLOCK,
                                SHARE_MEM_INDEX_TYPE, BP_INDEX_TYPE, BP_DATA_TYPE, SCALE_DATA_TYPE, A_TYPE,
                                COMPUTE_MULTI_TYPE, COMPUTE_SUM_TYPE, X_TYPE, Y_TYPE, NNZ_TYPE, ROW_TYPE,
                                COLUMN_TYPE>)<<<(BLOCKS), THREADS_PER_BLOCK, 0, stream>>>(
      m, alpha, beta, rowptr, break_points, colindex, value, x, y);
  USPARSE_HIP_API_CHECK(hipFreeAsync(break_points, stream));
}

template <typename HIP_BLOCK_NUM_TYPE, typename SHARE_MEM_INDEX_TYPE, typename BP_INDEX_TYPE, typename BP_DATA_TYPE,
          typename A_TYPE, typename X_TYPE, typename Y_TYPE, typename COMPUTE_MULTI_TYPE, typename COMPUTE_SUM_TYPE,
          typename SCALE_DATA_TYPE, typename NNZ_TYPE, typename ROW_TYPE, typename COLUMN_TYPE, int R,
          int REDUCE_OPTION, int REDUCE_VEC_SIZE, int THREADS_PER_BLOCK, int FLAT_STORE_Y_POLICY>
inline void flat_one_pass_sparse_spmv(hipStream_t stream, int trans, const SCALE_DATA_TYPE alpha,
                                      const SCALE_DATA_TYPE beta, ROW_TYPE m, COLUMN_TYPE n, NNZ_TYPE nnz,
                                      const NNZ_TYPE *rowptr, const COLUMN_TYPE *colindex, const A_TYPE *value,
                                      const X_TYPE *x, Y_TYPE *y, const BP_DATA_TYPE *break_points,
                                      KV<ROW_TYPE, Y_TYPE> *key_val) {
  // each block can process `R*THREADS_PER_BLOCK` non-zeros.
  const HIP_BLOCK_NUM_TYPE HIP_BLOCKS = nnz / (R * THREADS_PER_BLOCK) + ((nnz % (R * THREADS_PER_BLOCK) == 0) ? 0 : 1);

  (csrmv_flat_one_pass_kernel<FLAT_STORE_Y_POLICY, __WF_SIZE__, R, REDUCE_OPTION, REDUCE_VEC_SIZE, THREADS_PER_BLOCK,
                              SHARE_MEM_INDEX_TYPE, BP_INDEX_TYPE, BP_DATA_TYPE, SCALE_DATA_TYPE, A_TYPE,
                              COMPUTE_MULTI_TYPE, COMPUTE_SUM_TYPE, X_TYPE, Y_TYPE, NNZ_TYPE, ROW_TYPE,
                              COLUMN_TYPE>)<<<(HIP_BLOCKS), (THREADS_PER_BLOCK), 0, stream>>>(
      m, alpha, beta, rowptr, break_points, colindex, value, x, y, key_val);

  if (FLAT_STORE_Y_POLICY == FLAT_STORE_Y_POLICY_SEGMENTED_SUM) {
    if (key_val == nullptr) {
      // todo: return with error
      return;
    }
    // reduce by key. value: y_part_sum, key: y_part_indices, result: y.
    constexpr int CARRY_OUT_THREADS_PER_BLOCK = THREADS_PER_BLOCK;
    flat_reduce_carry_out<ROW_TYPE, Y_TYPE, HIP_BLOCK_NUM_TYPE, CARRY_OUT_THREADS_PER_BLOCK>(stream, y, key_val,
                                                                                             HIP_BLOCKS);
  }
  return;
}

template <typename DATA_TYPE, typename NNZ_TYPE, typename ROW_TYPE, typename COLUMN_TYPE, typename BP_DATA_TYPE,
          typename Y_TYPE, int FLAT_STORE_Y_POLICY, bool CREAT_PART_SUM_TEMP_BUFFER, int R, int THREADS_PER_BLOCK>
void flat_matrix_analyse_imp(hipStream_t stream,
                             const csr_mat_descr<DATA_TYPE, NNZ_TYPE, ROW_TYPE, COLUMN_TYPE> d_csr_desc,
                             const csrmv_flat_mtx_info<BP_DATA_TYPE, Y_TYPE> *info) {
  const NNZ_TYPE nnz = d_csr_desc.nnz;
  const ROW_TYPE m = d_csr_desc.rows;

  const int HIP_BLOCKS = nnz / (R * THREADS_PER_BLOCK) + ((nnz % (R * THREADS_PER_BLOCK) == 0) ? 0 : 1);
  constexpr int TOTAL_ROUNDS = 1;
  const NNZ_TYPE break_points_len = static_cast<NNZ_TYPE>(TOTAL_ROUNDS * HIP_BLOCKS + 1);

  USPARSE_HIP_API_CHECK(
      hipMallocAsync((void **)(&(info->break_points)), break_points_len * sizeof(BP_DATA_TYPE), stream));
  USPARSE_HIP_API_CHECK(hipMemsetAsync(info->break_points, 0, break_points_len * sizeof(BP_DATA_TYPE), stream));
  (pre_calc_break_point<ROW_TYPE, NNZ_TYPE, BP_DATA_TYPE, R * THREADS_PER_BLOCK>)<<<1024, 512, 0, stream>>>(
      d_csr_desc.row_ptr, m, info->break_points, break_points_len);

  // create temp buffer for storing part sum of y.
  if (FLAT_STORE_Y_POLICY == FLAT_STORE_Y_POLICY_SEGMENTED_SUM && CREAT_PART_SUM_TEMP_BUFFER) {
    // size: 2 * (HIP_BLOCKS + 1). Note: 2x is used for reduction-by-key recursively.
    USPARSE_HIP_API_CHECK(
        hipMallocAsync((void **)(&(info->key_val)), 2 * (HIP_BLOCKS + 1) * sizeof(KV<ROW_TYPE, Y_TYPE>), stream));
    USPARSE_HIP_API_CHECK(
        hipMemsetAsync(info->key_val, 0, 2 * (HIP_BLOCKS + 1) * sizeof(KV<ROW_TYPE, Y_TYPE>), stream));
  }
}

constexpr int FLAT_STAGE_ANALYSE = 0;
constexpr int FLAT_STAGE_COMPUTE = 1;
constexpr int FLAT_STAGE_DESTROY = 2;

/**
 * run different stage of the flat in one function, which can avoid code duplication of different stage calling and make
 * the logic clearer. In this function, we can set the compile-time parameter via template paramter.
 * @tparam T the compute data type.
 * @tparam DATA_TYPE the data type of the matrix value.
 * @tparam STAGE current compute stage: can be `FLAT_STAGE_ANALYSE`, `FLAT_STAGE_COMPUTE` or `FLAT_STAGE_DESTROY`.
 *
 * @param stage: the current stage of the flat, can be FLAT_STAGE_ANALYSE, FLAT_STAGE_COMPUTE or FLAT_STAGE_DESTROY.
 * @param stream the hip stream used for launching kernel and memory operations.
 * @param d_csr_desc the csr matrix descriptor.
 * @param info pointer to the matrix analyse info.
 * @param trans matrix transpose.
 * @param alpha the alpha in SpMV: y = alpha * A * x + beta * y.
 * @param beta the alpha in SpMV: y = alpha * A * x + beta * y.
 * @param x the vector x in SpMV: y = alpha * A * x + beta * y.
 * @param y the vector y in SpMV: y = alpha * A * x + beta * y.
 * @note: for the analyse stage. @param trans, alpha, beta, x and y can be ignored.
 *  for the compute stage, the above parameters should be set based on the csrmv input.
 *  for the destroy stage, all parameters (except @param stage, info, stream) can be ignored.
 */
template <typename T, typename NNZ_TYPE, typename ROW_TYPE, typename COLUMN_TYPE, typename BP_DATA_TYPE,
          typename DATA_TYPE, typename X_TYPE, typename Y_TYPE, typename SCALE_TYPE, int STAGE, int FLAT_STORE_Y_POLICY,
          bool CREAT_PART_SUM_TEMP_BUFFER, int R, int THREADS_PER_BLOCK, int FLAT_REDUCE_VEC_SIZE>
inline void flat_run_stage_template(hipStream_t stream,
                                    const csr_mat_descr<DATA_TYPE, NNZ_TYPE, ROW_TYPE, COLUMN_TYPE> d_csr_desc,
                                    const csrmv_flat_mtx_info<BP_DATA_TYPE, Y_TYPE> *info, int trans,
                                    const SCALE_TYPE alpha, const SCALE_TYPE beta, const X_TYPE *x, Y_TYPE *y) {
  constexpr bool __stage_check =
      (STAGE == FLAT_STAGE_ANALYSE || STAGE == FLAT_STAGE_COMPUTE || STAGE == FLAT_STAGE_DESTROY);
  static_assert(__stage_check,
                "the stage should be `FLAT_STAGE_ANALYSE`, `FLAT_STAGE_COMPUTE` or `FLAT_STAGE_DESTROY`.");

  EXPAND_FROM_CSR_MAT_DESCR(DATA_TYPE, NNZ_TYPE, ROW_TYPE, COLUMN_TYPE, d_csr_desc)

  if constexpr (STAGE == FLAT_STAGE_ANALYSE) {
    flat_matrix_analyse_imp<DATA_TYPE, NNZ_TYPE, ROW_TYPE, COLUMN_TYPE, BP_DATA_TYPE, Y_TYPE, FLAT_STORE_Y_POLICY,
                            CREAT_PART_SUM_TEMP_BUFFER, R, THREADS_PER_BLOCK>(stream, d_csr_desc, info);
  } else if (STAGE == FLAT_STAGE_COMPUTE) {
    using COMPUTE_MULTI_TYPE = T;
    using COMPUTE_SUM_TYPE = T;

    using HIP_BLOCK_NUM_TYPE = int;
    using SHARE_MEM_INDEX_TYPE = int;
    using BP_INDEX_TYPE = int;

    // todo: if y_part_sum is null. malloc device memory.

    BP_DATA_TYPE *break_points = info->break_points;
    KV<ROW_TYPE, Y_TYPE> *key_val = info->key_val;
    // Note: memset to 0 for y_part_sum and y_part_indices before the kernel launching is unnecessary.
    // because the csrmv kernel will write result using `y_part_sum[i] = val`, instead of `y_part_sum[i] += val`.

    constexpr int FLAT_REDUCE_OPTION = FLAT_REDUCE_VEC_SIZE == 1 ? FLAT_REDUCE_OPTION_DIRECT : FLAT_REDUCE_OPTION_VEC;
    flat_one_pass_sparse_spmv<HIP_BLOCK_NUM_TYPE, SHARE_MEM_INDEX_TYPE, BP_INDEX_TYPE, BP_DATA_TYPE, DATA_TYPE, X_TYPE,
                              Y_TYPE, COMPUTE_MULTI_TYPE, COMPUTE_SUM_TYPE, SCALE_TYPE, NNZ_TYPE, ROW_TYPE, COLUMN_TYPE,
                              R, FLAT_REDUCE_OPTION, FLAT_REDUCE_VEC_SIZE, THREADS_PER_BLOCK, FLAT_STORE_Y_POLICY>(
        stream, trans, alpha, beta, m, n, nnz, rowptr, colindex, value, x, y, break_points, key_val);
  } else if (STAGE == FLAT_STAGE_DESTROY) {
    USPARSE_HIP_API_CHECK(hipFreeAsync(info->break_points, stream));
    if (FLAT_STORE_Y_POLICY == FLAT_STORE_Y_POLICY_SEGMENTED_SUM) {
      USPARSE_HIP_API_CHECK(hipFreeAsync(info->key_val, stream));
    }
  } else {
    // return
  }
}

/**
 * adaptively select compile-time parameter based on the matrix characteristics for analyse, compute and destroy stage.
 *
 * Note: the matrix analyse and computer stages should all call this function.
 * Because matrix analyse and compute shoule pass the same R, THREADS_PER_BLOCK and FLAT_REDUCE_VEC_SIZE for the same
 * matrix. we use this adaptive parameter selection routine to select the compile-time parameter for both stages to
 * avoid code duplication.
 */
template <typename T, typename NNZ_TYPE, typename ROW_TYPE, typename COLUMN_TYPE, typename BP_DATA_TYPE, typename A,
          typename X_TYPE, typename Y_TYPE, typename SCALE_TYPE, int STAGE, int FLAT_STORE_Y_POLICY,
          bool CREAT_PART_SUM_TEMP_BUFFER>
inline void adaptive_flat_param_dispatch_for_stages(hipStream_t stream,
                                                    const csr_mat_descr<A, NNZ_TYPE, ROW_TYPE, COLUMN_TYPE> d_csr_desc,
                                                    const csrmv_flat_mtx_info<BP_DATA_TYPE, Y_TYPE> *info,
                                                    const int trans, const SCALE_TYPE alpha, const SCALE_TYPE beta,
                                                    const X_TYPE *x, Y_TYPE *y) {
  EXPAND_FROM_CSR_MAT_DESCR(A, NNZ_TYPE, ROW_TYPE, COLUMN_TYPE, d_csr_desc)

#define FLAT_RUN_STAGE_WITH_PARAM(R, THREADS_PER_BLOCK, FLAT_REDUCE_VEC_SIZE)                                          \
  flat_run_stage_template<T, NNZ_TYPE, ROW_TYPE, COLUMN_TYPE, BP_DATA_TYPE, A, X_TYPE, Y_TYPE, SCALE_TYPE, STAGE,      \
                          FLAT_STORE_Y_POLICY, CREAT_PART_SUM_TEMP_BUFFER, R, THREADS_PER_BLOCK,                       \
                          FLAT_REDUCE_VEC_SIZE>(stream, d_csr_desc, info, trans, alpha, beta, x, y);

  // set different compile-time parameter (R, THREADS_PER_BLOCK and FLAT_REDUCE_VEC_SIZE) based on the matrix
  // characteristics.
  NNZ_TYPE avg_block_nnz_max = nnz / m;

  if (avg_block_nnz_max <= 4) {
    // Extreme sparse class: maximize average speedup
    FLAT_RUN_STAGE_WITH_PARAM(2, 256, 4);
  } else if (avg_block_nnz_max <= 32) {
    // Sparse to moderate classes: 512 threads with VEC_SIZE 8 yields the highest average acceleration
    FLAT_RUN_STAGE_WITH_PARAM(2, 512, 8);
  } else if (avg_block_nnz_max <= 64) {
    // Shift to R=4 to mask latency for higher density
    FLAT_RUN_STAGE_WITH_PARAM(4, 512, 8);
  } else if (avg_block_nnz_max <= 128) {
    constexpr int REDUCE_OPT = FLAT_REDUCE_OPTION_VEC;
    FLAT_RUN_STAGE_WITH_PARAM(4, 512, 16);
  } else if (avg_block_nnz_max <= 256) {
    constexpr int REDUCE_OPT = FLAT_REDUCE_OPTION_VEC;
    FLAT_RUN_STAGE_WITH_PARAM(4, 512, 32);
  } else {
    // Highly dense class
    constexpr int REDUCE_OPT = FLAT_REDUCE_OPTION_VEC;
    FLAT_RUN_STAGE_WITH_PARAM(4, 512, 64);
  }
}

template <typename DATA_TYPE, typename NNZ_TYPE, typename ROW_TYPE, typename COLUMN_TYPE, typename BP_DATA_TYPE,
          typename Y_TYPE, int FLAT_STORE_Y_POLICY, bool CREAT_PART_SUM_TEMP_BUFFER>
void flat_matrix_analyse(hipStream_t stream, const csr_mat_descr<DATA_TYPE, NNZ_TYPE, ROW_TYPE, COLUMN_TYPE> d_csr_desc,
                         csrmv_flat_mtx_info<BP_DATA_TYPE, Y_TYPE> *info) {
  using T = DATA_TYPE;               // T is not used in the analyse stage.
  using SCALE_DATA_TYPE = DATA_TYPE; // SCALE_DATA_TYPE is not used in the analyse stage.
  using X_TYPE = Y_TYPE;             // X_TYPE is not used in the analyse stage.
  constexpr int trans = 0;           // trans is not used in the analyse stage.

  adaptive_flat_param_dispatch_for_stages<T, NNZ_TYPE, ROW_TYPE, COLUMN_TYPE, BP_DATA_TYPE, DATA_TYPE, X_TYPE, Y_TYPE,
                                          SCALE_DATA_TYPE, FLAT_STAGE_ANALYSE, FLAT_STORE_Y_POLICY,
                                          CREAT_PART_SUM_TEMP_BUFFER>(
      stream, d_csr_desc, info, trans, static_cast<SCALE_DATA_TYPE>(0.0), static_cast<SCALE_DATA_TYPE>(0.0), nullptr,
      nullptr);
}

template <typename DATA_TYPE, typename NNZ_TYPE, typename ROW_TYPE, typename COLUMN_TYPE, typename BP_DATA_TYPE,
          typename Y_TYPE>
void flat_matrix_analyse_atomic(hipStream_t stream,
                                const csr_mat_descr<DATA_TYPE, NNZ_TYPE, ROW_TYPE, COLUMN_TYPE> d_csr_desc,
                                csrmv_flat_mtx_info<BP_DATA_TYPE, Y_TYPE> *info) {
  flat_matrix_analyse<DATA_TYPE, NNZ_TYPE, ROW_TYPE, COLUMN_TYPE, BP_DATA_TYPE, Y_TYPE, FLAT_STORE_Y_POLICY_ATOMIC,
                      false>(stream, d_csr_desc, info);
  return;
}

template <typename DATA_TYPE, typename NNZ_TYPE, typename ROW_TYPE, typename COLUMN_TYPE, typename BP_DATA_TYPE,
          typename Y_TYPE>
void flat_matrix_analyse_segmented(hipStream_t stream,
                                   const csr_mat_descr<DATA_TYPE, NNZ_TYPE, ROW_TYPE, COLUMN_TYPE> d_csr_desc,
                                   csrmv_flat_mtx_info<BP_DATA_TYPE, Y_TYPE> *info) {
  flat_matrix_analyse<DATA_TYPE, NNZ_TYPE, ROW_TYPE, COLUMN_TYPE, BP_DATA_TYPE, Y_TYPE,
                      FLAT_STORE_Y_POLICY_SEGMENTED_SUM, true>(stream, d_csr_desc, info);
  return;
}

template <typename BP_DATA_TYPE, typename Y_TYPE, int FLAT_STORE_Y_POLICY, bool CREAT_PART_SUM_TEMP_BUFFER>
void flat_matrix_clear(hipStream_t stream, csrmv_flat_mtx_info<BP_DATA_TYPE, Y_TYPE> *info) {
  USPARSE_HIP_API_CHECK(hipFree(info->break_points));
  if (FLAT_STORE_Y_POLICY == FLAT_STORE_Y_POLICY_SEGMENTED_SUM && CREAT_PART_SUM_TEMP_BUFFER) {
    USPARSE_HIP_API_CHECK(hipFree(info->key_val));
  }
}

template <typename BP_DATA_TYPE, typename Y_TYPE>
void flat_matrix_clear_atomic(hipStream_t stream, csrmv_flat_mtx_info<BP_DATA_TYPE, Y_TYPE> *info) {
  flat_matrix_clear<BP_DATA_TYPE, Y_TYPE, FLAT_STORE_Y_POLICY_ATOMIC, false>(stream, info);
  return;
}

template <typename BP_DATA_TYPE, typename Y_TYPE>
void flat_matrix_clear_segmented(hipStream_t stream, csrmv_flat_mtx_info<BP_DATA_TYPE, Y_TYPE> *info) {
  flat_matrix_clear<BP_DATA_TYPE, Y_TYPE, FLAT_STORE_Y_POLICY_SEGMENTED_SUM, true>(stream, info);
  return;
}

void flat_matrix_clear(hipStream_t stream, void *info) {
  // todo: usparse_default_flat_mtx_info is not elegant.
  usparse_default_flat_mtx_info *info_ptr = reinterpret_cast<usparse_default_flat_mtx_info *>(info);
  USPARSE_HIP_API_CHECK(hipFree(info_ptr->break_points));
  USPARSE_HIP_API_CHECK(hipFree(info_ptr->key_val));
}

void flat_matrix_clear_atomic(hipStream_t stream, void *info) {
  // todo: usparse_default_flat_mtx_info is not elegant.
  flat_matrix_clear_atomic(stream, reinterpret_cast<usparse_default_flat_mtx_info *>(info));
}

void flat_matrix_clear_segmented(hipStream_t stream, void *info) {
  // todo: usparse_default_flat_mtx_info is not elegant.
  flat_matrix_clear_segmented(stream, reinterpret_cast<usparse_default_flat_mtx_info *>(info));
}

/**
 * note: sugment-sum does not support FLAT_STORE_Y_POLICY_SEGMENTED_SUM mode currently.
 */
template <typename INDEX_TYPE, typename DATA_TYPE>
void segment_sum_flat_sparse_spmv(int trans, const DATA_TYPE alpha, const DATA_TYPE beta,
                                  const csr_mat_descr<DATA_TYPE, INDEX_TYPE, INDEX_TYPE, INDEX_TYPE> h_csr_desc,
                                  const csr_mat_descr<DATA_TYPE, INDEX_TYPE, INDEX_TYPE, INDEX_TYPE> d_csr_desc,
                                  const DATA_TYPE *x, DATA_TYPE *y) {
  EXPAND_FROM_CSR_MAT_DESCR(DATA_TYPE, INDEX_TYPE, INDEX_TYPE, INDEX_TYPE, d_csr_desc)

  constexpr int R = 2;
  constexpr int THREADS_PER_BLOCK = 512;
  constexpr int blocks = 512;
  constexpr int RED_OPT = FLAT_REDUCE_OPTION_SEGMENT_SUM;

  hipStream_t stream = hipStreamDefault;
  flat_multi_pass_sparse_spmv<INDEX_TYPE, INDEX_TYPE, INDEX_TYPE, INDEX_TYPE, DATA_TYPE, DATA_TYPE, DATA_TYPE,
                              DATA_TYPE, DATA_TYPE, DATA_TYPE, INDEX_TYPE, INDEX_TYPE, INDEX_TYPE, R, RED_OPT, 2,
                              blocks, THREADS_PER_BLOCK>(stream, trans, alpha, beta, m, n, nnz, rowptr, colindex, value,
                                                         x, y);
}

template <typename Y_TYPE, typename SCALE_DATA_TYPE, typename INDEX_TYPE, int BETA_Y_POLICY>
inline void flat_set_beta_y(hipStream_t stream, Y_TYPE *y, const INDEX_TYPE m, const SCALE_DATA_TYPE beta) {
  if (BETA_Y_POLICY == BETA_Y_POLICY_AUTO) {
    // set beta*y.
    if (beta == static_cast<SCALE_DATA_TYPE>(0.0)) {
      // set vector y to 0. because: y = A*x + 0.0 *y.
      usparse::device_scale_set<INDEX_TYPE, Y_TYPE>(static_cast<Y_TYPE>(0.0), y, m, stream);
    } else if (beta == static_cast<SCALE_DATA_TYPE>(1.0)) {
      // do nothing. because flat is based on atomicAdd.
    } else {
      usparse::device_dscal<INDEX_TYPE, Y_TYPE>(beta, y, m, stream);
    }
  } else if (BETA_Y_POLICY == BETA_Y_POLICY_ZERO) {
    usparse::device_scale_set<INDEX_TYPE, Y_TYPE>(static_cast<Y_TYPE>(0.0), y, m, stream);
  } else if (BETA_Y_POLICY == BETA_Y_POLICY_ONE) {
    return; // nothing to do.
  } else {  // BETA_Y_POLICY == BETA_Y_POLICY_OTHER
    usparse::device_dscal<INDEX_TYPE, Y_TYPE>(beta, y, m, stream);
  }
}

/**
 * set flat kernel template parameters adaptively.
 * currently, it only support adaptive template parameters on one-pass flat method.
 * \note: if config item FLAT_ONE_PASS_ADAPTIVE is set to false, one-pass adaptive will be disabled.
 */
template <typename T, typename BP_DATA_TYPE, typename A, typename X, typename Y, typename NNZ, typename ROW,
          typename COL, typename SCALE_TYPE, int BETA_Y_POLICY, int FLAT_STORE_Y_POLICY>
void adaptive_csrmv_flat(hipStream_t stream, int trans, const SCALE_TYPE alpha, const SCALE_TYPE beta,
                               const csr_mat_descr<A, NNZ, ROW, COL> d_csr_desc, const X *x, Y *y,
                               const csrmv_flat_mtx_info<BP_DATA_TYPE, Y> *info) {
  flat_set_beta_y<Y, SCALE_TYPE, ROW, BETA_Y_POLICY>(stream, y, d_csr_desc.rows, beta);

  EXPAND_FROM_CSR_MAT_DESCR(A, NNZ, ROW, COL, d_csr_desc);
  BP_DATA_TYPE *break_points = info->break_points;
  KV<ROW, Y> *key_val = info->key_val;
  using COMPUTE_MULTI_TYPE = T;
  using COMPUTE_SUM_TYPE = T;

  using HIP_BLOCK_NUM_TYPE = int;
  using SHARE_MEM_INDEX_TYPE = int;
  using BP_INDEX_TYPE = int;
  // for non one-pass.
  if (!FLAT_ONE_PASS) {
    constexpr int R = 2;
    constexpr int THREADS_PER_BLOCK = 512;
    constexpr HIP_BLOCK_NUM_TYPE blocks = 512;
    constexpr int RED_OPT = FLAT_REDUCE_OPTION_VEC;

    flat_multi_pass_sparse_spmv<HIP_BLOCK_NUM_TYPE, SHARE_MEM_INDEX_TYPE, BP_INDEX_TYPE, BP_DATA_TYPE, A, X, Y,
                                COMPUTE_MULTI_TYPE, COMPUTE_SUM_TYPE, SCALE_TYPE, NNZ, ROW, COL, R, RED_OPT, 2, blocks,
                                THREADS_PER_BLOCK>(stream, trans, alpha, beta, m, n, nnz, rowptr, colindex, value, x,
                                                   y);
    return;
  }

  // todo: if y_part_sum is null. malloc device memory.

  if (!FLAT_ONE_PASS_ADAPTIVE) {
    constexpr int R = 2;
    constexpr int THREADS_PER_BLOCK = 512;
    constexpr int RED_OPT = FLAT_REDUCE_OPTION_VEC;
    constexpr int VEC_SIZE = 4;
    flat_one_pass_sparse_spmv<HIP_BLOCK_NUM_TYPE, SHARE_MEM_INDEX_TYPE, BP_INDEX_TYPE, BP_DATA_TYPE, A, X, Y,
                              COMPUTE_MULTI_TYPE, COMPUTE_SUM_TYPE, SCALE_TYPE, NNZ, ROW, COL, R, RED_OPT, VEC_SIZE,
                              THREADS_PER_BLOCK, FLAT_STORE_Y_POLICY>(stream, trans, alpha, beta, m, n, nnz, rowptr,
                                                                      colindex, value, x, y, break_points, key_val);
    return;
  }

  // use adaptive one-pass.
  constexpr bool create_temp_buffer = false; // create_temp_buffer is not used in the compute stage.
  adaptive_flat_param_dispatch_for_stages<T, NNZ, ROW, COL, BP_DATA_TYPE, A, X, Y, SCALE_TYPE, FLAT_STAGE_COMPUTE,
                                          FLAT_STORE_Y_POLICY, create_temp_buffer>(stream, d_csr_desc, info, trans,
                                                                                   alpha, beta, x, y);
  return;
}

template <typename I, typename T, int BETA_Y_POLICY>
void adaptive_flat_sparse_spmv(int trans, const T alpha, const T beta, const csr_mat_descr<T, I, I, I> d_csr_desc,
                               const T *x, T *y, const csrmv_flat_mtx_info<I, T> *info) {
  hipStream_t stream = hipStreamDefault;
  adaptive_csrmv_flat_atomic<T, I, T, T, T, I, I, I, T, BETA_Y_POLICY>(stream, trans, alpha, beta, d_csr_desc, x, y,
                                                                      info);
}

template <typename T, typename BP_DATA_TYPE, typename A, typename X, typename Y, typename NNZ, typename ROW,
          typename COL, typename SCALE_TYPE, int BETA_Y_POLICY>
void adaptive_csrmv_flat_atomic(hipStream_t stream, int trans, const SCALE_TYPE alpha, const SCALE_TYPE beta,
                                const csr_mat_descr<A, NNZ, ROW, COL> d_csr_desc, const X *x, Y *y,
                                const csrmv_flat_mtx_info<BP_DATA_TYPE, Y> *info) {
  adaptive_csrmv_flat<T, BP_DATA_TYPE, A, X, Y, NNZ, ROW, COL, SCALE_TYPE, BETA_Y_POLICY, FLAT_STORE_Y_POLICY_ATOMIC>(
      stream, trans, alpha, beta, d_csr_desc, x, y, info);
  return;
}

template <typename T, typename BP_DATA_TYPE, typename A, typename X, typename Y, typename NNZ, typename ROW,
          typename COL, typename SCALE_TYPE, int BETA_Y_POLICY>
void adaptive_csrmv_flat_segmented(hipStream_t stream, int trans, const SCALE_TYPE alpha, const SCALE_TYPE beta,
                                   const csr_mat_descr<A, NNZ, ROW, COL> d_csr_desc, const X *x, Y *y,
                                   const csrmv_flat_mtx_info<BP_DATA_TYPE, Y> *info) {
  adaptive_csrmv_flat<T, BP_DATA_TYPE, A, X, Y, NNZ, ROW, COL, SCALE_TYPE, BETA_Y_POLICY,
                      FLAT_STORE_Y_POLICY_SEGMENTED_SUM>(stream, trans, alpha, beta, d_csr_desc, x, y, info);
  return;
}

#define FLAT_SPMV_INSTANTIATE_MIXED(T, BP, A, X, Y, NNZ, ROW, COL, SCALE)                                              \
  template void adaptive_csrmv_flat_atomic<T, BP, A, X, Y, NNZ, ROW, COL, SCALE, BETA_Y_POLICY_AUTO>(                  \
      hipStream_t stream, int trans, const SCALE alpha, const SCALE beta,                                              \
      const csr_mat_descr<A, NNZ, ROW, COL> d_csr_desc, const X *x, Y *y, const csrmv_flat_mtx_info<BP, Y> *info);     \
  template void adaptive_csrmv_flat_atomic<T, BP, A, X, Y, NNZ, ROW, COL, SCALE, BETA_Y_POLICY_ZERO>(                  \
      hipStream_t stream, int trans, const SCALE alpha, const SCALE beta,                                              \
      const csr_mat_descr<A, NNZ, ROW, COL> d_csr_desc, const X *x, Y *y, const csrmv_flat_mtx_info<BP, Y> *info);     \
  template void adaptive_csrmv_flat_atomic<T, BP, A, X, Y, NNZ, ROW, COL, SCALE, BETA_Y_POLICY_ONE>(                   \
      hipStream_t stream, int trans, const SCALE alpha, const SCALE beta,                                              \
      const csr_mat_descr<A, NNZ, ROW, COL> d_csr_desc, const X *x, Y *y, const csrmv_flat_mtx_info<BP, Y> *info);     \
  template void adaptive_csrmv_flat_atomic<T, BP, A, X, Y, NNZ, ROW, COL, SCALE, BETA_Y_POLICY_OTHER>(                 \
      hipStream_t stream, int trans, const SCALE alpha, const SCALE beta,                                              \
      const csr_mat_descr<A, NNZ, ROW, COL> d_csr_desc, const X *x, Y *y, const csrmv_flat_mtx_info<BP, Y> *info);     \
                                                                                                                       \
  template void adaptive_csrmv_flat_segmented<T, BP, A, X, Y, NNZ, ROW, COL, SCALE, BETA_Y_POLICY_AUTO>(               \
      hipStream_t stream, int trans, const SCALE alpha, const SCALE beta,                                              \
      const csr_mat_descr<A, NNZ, ROW, COL> d_csr_desc, const X *x, Y *y, const csrmv_flat_mtx_info<BP, Y> *info);     \
  template void adaptive_csrmv_flat_segmented<T, BP, A, X, Y, NNZ, ROW, COL, SCALE, BETA_Y_POLICY_ZERO>(               \
      hipStream_t stream, int trans, const SCALE alpha, const SCALE beta,                                              \
      const csr_mat_descr<A, NNZ, ROW, COL> d_csr_desc, const X *x, Y *y, const csrmv_flat_mtx_info<BP, Y> *info);     \
  template void adaptive_csrmv_flat_segmented<T, BP, A, X, Y, NNZ, ROW, COL, SCALE, BETA_Y_POLICY_ONE>(                \
      hipStream_t stream, int trans, const SCALE alpha, const SCALE beta,                                              \
      const csr_mat_descr<A, NNZ, ROW, COL> d_csr_desc, const X *x, Y *y, const csrmv_flat_mtx_info<BP, Y> *info);     \
  template void adaptive_csrmv_flat_segmented<T, BP, A, X, Y, NNZ, ROW, COL, SCALE, BETA_Y_POLICY_OTHER>(              \
      hipStream_t stream, int trans, const SCALE alpha, const SCALE beta,                                              \
      const csr_mat_descr<A, NNZ, ROW, COL> d_csr_desc, const X *x, Y *y, const csrmv_flat_mtx_info<BP, Y> *info);

FLAT_SPMV_INSTANTIATE_MIXED(usparse::usparse_fp64, usparse::usparse_int, usparse::usparse_fp64, usparse::usparse_fp64,
                            usparse::usparse_fp64, usparse::usparse_int, usparse::usparse_int, usparse::usparse_int,
                            usparse::usparse_fp64)
FLAT_SPMV_INSTANTIATE_MIXED(usparse::usparse_fp32, usparse::usparse_int, usparse::usparse_fp32, usparse::usparse_fp32,
                            usparse::usparse_fp32, usparse::usparse_int, usparse::usparse_int, usparse::usparse_int,
                            usparse::usparse_fp32)
FLAT_SPMV_INSTANTIATE_MIXED(usparse::usparse_fpc32, usparse::usparse_int, usparse::usparse_fpc32,
                            usparse::usparse_fpc32, usparse::usparse_fpc32, usparse::usparse_int, usparse::usparse_int,
                            usparse::usparse_int, usparse::usparse_fpc32)
FLAT_SPMV_INSTANTIATE_MIXED(usparse::usparse_fpc64, usparse::usparse_int, usparse::usparse_fpc64,
                            usparse::usparse_fpc64, usparse::usparse_fpc64, usparse::usparse_int, usparse::usparse_int,
                            usparse::usparse_int, usparse::usparse_fpc64)

FLAT_SPMV_INSTANTIATE_MIXED(int32_t, int32_t, int8_t, int8_t, int32_t, int32_t, int32_t, int32_t, int32_t)
FLAT_SPMV_INSTANTIATE_MIXED(int32_t, int32_t, int8_t, int8_t, int32_t, int64_t, int32_t, int32_t, int32_t)
FLAT_SPMV_INSTANTIATE_MIXED(int32_t, int64_t, int8_t, int8_t, int32_t, int64_t, int64_t, int64_t, int32_t)

FLAT_SPMV_INSTANTIATE_MIXED(usparse::usparse_fp32, int32_t, int8_t, int8_t, usparse::usparse_fp32, int32_t, int32_t,
                            int32_t, usparse::usparse_fp32)
FLAT_SPMV_INSTANTIATE_MIXED(usparse::usparse_fp64, int32_t, int8_t, int8_t, usparse::usparse_fp64, int32_t, int32_t,
                            int32_t, usparse::usparse_fp64)
FLAT_SPMV_INSTANTIATE_MIXED(usparse::usparse_fpc32, int32_t, int8_t, int8_t, usparse::usparse_fpc32, int32_t, int32_t,
                            int32_t, usparse::usparse_fpc32)
FLAT_SPMV_INSTANTIATE_MIXED(usparse::usparse_fpc64, int32_t, int8_t, int8_t, usparse::usparse_fpc64, int32_t, int32_t,
                            int32_t, usparse::usparse_fpc64)

FLAT_SPMV_INSTANTIATE_MIXED(usparse::usparse_fp32, int32_t, int8_t, int8_t, usparse::usparse_fp32, int64_t, int32_t,
                            int32_t, usparse::usparse_fp32)
FLAT_SPMV_INSTANTIATE_MIXED(usparse::usparse_fp32, int64_t, int8_t, int8_t, usparse::usparse_fp32, int64_t, int64_t,
                            int64_t, usparse::usparse_fp32)
FLAT_SPMV_INSTANTIATE_MIXED(usparse::usparse_fp64, int32_t, usparse::usparse_fp32, usparse::usparse_fp64,
                            usparse::usparse_fp64, int32_t, int32_t, int32_t, usparse::usparse_fp64)
FLAT_SPMV_INSTANTIATE_MIXED(usparse::usparse_fp64, int32_t, usparse::usparse_fp32, usparse::usparse_fp64,
                            usparse::usparse_fp64, int64_t, int32_t, int32_t, usparse::usparse_fp64)
FLAT_SPMV_INSTANTIATE_MIXED(usparse::usparse_fp64, int64_t, usparse::usparse_fp32, usparse::usparse_fp64,
                            usparse::usparse_fp64, int64_t, int64_t, int64_t, usparse::usparse_fp64)

FLAT_SPMV_INSTANTIATE_MIXED(usparse::usparse_fpc32, int32_t, usparse::usparse_fp32, usparse::usparse_fpc32,
                            usparse::usparse_fpc32, int32_t, int32_t, int32_t, usparse::usparse_fpc32)
FLAT_SPMV_INSTANTIATE_MIXED(usparse::usparse_fpc32, int32_t, usparse::usparse_fp32, usparse::usparse_fpc32,
                            usparse::usparse_fpc32, int64_t, int32_t, int32_t, usparse::usparse_fpc32)
FLAT_SPMV_INSTANTIATE_MIXED(usparse::usparse_fpc32, int64_t, usparse::usparse_fp32, usparse::usparse_fpc32,
                            usparse::usparse_fpc32, int64_t, int64_t, int64_t, usparse::usparse_fpc32)

FLAT_SPMV_INSTANTIATE_MIXED(usparse::usparse_fpc64, int32_t, usparse::usparse_fpc32, usparse::usparse_fpc64,
                            usparse::usparse_fpc64, int32_t, int32_t, int32_t, usparse::usparse_fpc64)
FLAT_SPMV_INSTANTIATE_MIXED(usparse::usparse_fpc64, int32_t, usparse::usparse_fpc32, usparse::usparse_fpc64,
                            usparse::usparse_fpc64, int64_t, int32_t, int32_t, usparse::usparse_fpc64)
FLAT_SPMV_INSTANTIATE_MIXED(usparse::usparse_fpc64, int64_t, usparse::usparse_fpc32, usparse::usparse_fpc64,
                            usparse::usparse_fpc64, int64_t, int64_t, int64_t, usparse::usparse_fpc64)

FLAT_SPMV_INSTANTIATE_MIXED(usparse::usparse_fpc64, int32_t, usparse::usparse_fp64, usparse::usparse_fpc64,
                            usparse::usparse_fpc64, int32_t, int32_t, int32_t, usparse::usparse_fpc64)
FLAT_SPMV_INSTANTIATE_MIXED(usparse::usparse_fpc64, int32_t, usparse::usparse_fp64, usparse::usparse_fpc64,
                            usparse::usparse_fpc64, int64_t, int32_t, int32_t, usparse::usparse_fpc64)
FLAT_SPMV_INSTANTIATE_MIXED(usparse::usparse_fpc64, int64_t, usparse::usparse_fp64, usparse::usparse_fpc64,
                            usparse::usparse_fpc64, int64_t, int64_t, int64_t, usparse::usparse_fpc64)

FLAT_SPMV_INSTANTIATE_MIXED(usparse::usparse_fp32, int32_t, usparse::usparse_fp32, usparse::usparse_fp32,
                            usparse::usparse_fp32, int64_t, int32_t, int32_t, usparse::usparse_fp32)
FLAT_SPMV_INSTANTIATE_MIXED(usparse::usparse_fp32, int64_t, usparse::usparse_fp32, usparse::usparse_fp32,
                            usparse::usparse_fp32, int64_t, int64_t, int64_t, usparse::usparse_fp32)

FLAT_SPMV_INSTANTIATE_MIXED(usparse::usparse_fp64, int32_t, usparse::usparse_fp64, usparse::usparse_fp64,
                            usparse::usparse_fp64, int64_t, int32_t, int32_t, usparse::usparse_fp64)
FLAT_SPMV_INSTANTIATE_MIXED(usparse::usparse_fp64, int64_t, usparse::usparse_fp64, usparse::usparse_fp64,
                            usparse::usparse_fp64, int64_t, int64_t, int64_t, usparse::usparse_fp64)

FLAT_SPMV_INSTANTIATE_MIXED(usparse::usparse_fpc32, int32_t, usparse::usparse_fpc32, usparse::usparse_fpc32,
                            usparse::usparse_fpc32, int64_t, int32_t, int32_t, usparse::usparse_fpc32)
FLAT_SPMV_INSTANTIATE_MIXED(usparse::usparse_fpc32, int64_t, usparse::usparse_fpc32, usparse::usparse_fpc32,
                            usparse::usparse_fpc32, int64_t, int64_t, int64_t, usparse::usparse_fpc32)

FLAT_SPMV_INSTANTIATE_MIXED(usparse::usparse_fpc64, int32_t, usparse::usparse_fpc64, usparse::usparse_fpc64,
                            usparse::usparse_fpc64, int64_t, int32_t, int32_t, usparse::usparse_fpc64)
FLAT_SPMV_INSTANTIATE_MIXED(usparse::usparse_fpc64, int64_t, usparse::usparse_fpc64, usparse::usparse_fpc64,
                            usparse::usparse_fpc64, int64_t, int64_t, int64_t, usparse::usparse_fpc64)

#define FLAT_SPMV_INSTANTIATE_DEPRECATED(I, T)                                                                         \
  template void adaptive_flat_sparse_spmv<I, T, BETA_Y_POLICY_AUTO>(                                                   \
      int trans, const T alpha, const T beta, const csr_mat_descr<T, I, I, I> d_csr_desc, const T *x, T *y,            \
      const csrmv_flat_mtx_info<I, T> *info);                                                                          \
  template void adaptive_flat_sparse_spmv<I, T, BETA_Y_POLICY_ZERO>(                                                   \
      int trans, const T alpha, const T beta, const csr_mat_descr<T, I, I, I> d_csr_desc, const T *x, T *y,            \
      const csrmv_flat_mtx_info<I, T> *info);                                                                          \
  template void adaptive_flat_sparse_spmv<I, T, BETA_Y_POLICY_ONE>(                                                    \
      int trans, const T alpha, const T beta, const csr_mat_descr<T, I, I, I> d_csr_desc, const T *x, T *y,            \
      const csrmv_flat_mtx_info<I, T> *info);                                                                          \
  template void adaptive_flat_sparse_spmv<I, T, BETA_Y_POLICY_OTHER>(                                                  \
      int trans, const T alpha, const T beta, const csr_mat_descr<T, I, I, I> d_csr_desc, const T *x, T *y,            \
      const csrmv_flat_mtx_info<I, T> *info);

FLAT_SPMV_INSTANTIATE_DEPRECATED(usparse::usparse_int, usparse::usparse_fp64)
FLAT_SPMV_INSTANTIATE_DEPRECATED(usparse::usparse_int, usparse::usparse_fp32)
FLAT_SPMV_INSTANTIATE_DEPRECATED(usparse::usparse_int, usparse::usparse_fpc32)
FLAT_SPMV_INSTANTIATE_DEPRECATED(usparse::usparse_int, usparse::usparse_fpc64)

#define FLAT_SPMV_ANALYSE_INSTANTIATE(A, NNZ, ROW, COL, BP, Y)                                                         \
  template void flat_matrix_analyse_atomic<A, NNZ, ROW, COL, BP, Y>(                                                   \
      hipStream_t stream, const csr_mat_descr<A, NNZ, ROW, COL> d_csr_desc, csrmv_flat_mtx_info<BP, Y> *info);         \
  template void flat_matrix_analyse_segmented<A, NNZ, ROW, COL, BP, Y>(                                                \
      hipStream_t stream, const csr_mat_descr<A, NNZ, ROW, COL> d_csr_desc, csrmv_flat_mtx_info<BP, Y> *info);

FLAT_SPMV_ANALYSE_INSTANTIATE(usparse::usparse_fp64, usparse::usparse_int, usparse::usparse_int, usparse::usparse_int,
                              usparse::usparse_int, usparse::usparse_fp64);
FLAT_SPMV_ANALYSE_INSTANTIATE(usparse::usparse_fp32, usparse::usparse_int, usparse::usparse_int, usparse::usparse_int,
                              usparse::usparse_int, usparse::usparse_fp32);
FLAT_SPMV_ANALYSE_INSTANTIATE(usparse::usparse_fpc64, usparse::usparse_int, usparse::usparse_int, usparse::usparse_int,
                              usparse::usparse_int, usparse::usparse_fpc64);
FLAT_SPMV_ANALYSE_INSTANTIATE(usparse::usparse_fpc32, usparse::usparse_int, usparse::usparse_int, usparse::usparse_int,
                              usparse::usparse_int, usparse::usparse_fpc32);
FLAT_SPMV_ANALYSE_INSTANTIATE(int8_t, usparse::usparse_int, usparse::usparse_int, usparse::usparse_int,
                              usparse::usparse_int, int8_t);

FLAT_SPMV_ANALYSE_INSTANTIATE(int8_t, int64_t, int32_t, int32_t, int32_t, int8_t);
FLAT_SPMV_ANALYSE_INSTANTIATE(int8_t, int64_t, int64_t, int64_t, int64_t, int8_t);
FLAT_SPMV_ANALYSE_INSTANTIATE(usparse::usparse_fp32, int64_t, int32_t, int32_t, int32_t, usparse::usparse_fp32);
FLAT_SPMV_ANALYSE_INSTANTIATE(usparse::usparse_fp32, int64_t, int64_t, int64_t, int64_t, usparse::usparse_fp32);
FLAT_SPMV_ANALYSE_INSTANTIATE(usparse::usparse_fp64, int64_t, int32_t, int32_t, int32_t, usparse::usparse_fp64);
FLAT_SPMV_ANALYSE_INSTANTIATE(usparse::usparse_fp64, int64_t, int64_t, int64_t, int64_t, usparse::usparse_fp64);
FLAT_SPMV_ANALYSE_INSTANTIATE(usparse::usparse_fpc32, int64_t, int32_t, int32_t, int32_t, usparse::usparse_fpc32);
FLAT_SPMV_ANALYSE_INSTANTIATE(usparse::usparse_fpc32, int64_t, int64_t, int64_t, int64_t, usparse::usparse_fpc32);
FLAT_SPMV_ANALYSE_INSTANTIATE(usparse::usparse_fpc64, int64_t, int32_t, int32_t, int32_t, usparse::usparse_fpc64);
FLAT_SPMV_ANALYSE_INSTANTIATE(usparse::usparse_fpc64, int64_t, int64_t, int64_t, int64_t, usparse::usparse_fpc64);

#define FLAT_SPMV_CLEAR_INSTANTIATE(BP, Y)                                                                             \
  template void flat_matrix_clear_atomic<BP, Y>(hipStream_t stream, csrmv_flat_mtx_info<BP, Y> * info);                \
  template void flat_matrix_clear_segmented<BP, Y>(hipStream_t stream, csrmv_flat_mtx_info<BP, Y> * info);

FLAT_SPMV_CLEAR_INSTANTIATE(usparse::usparse_int, usparse::usparse_fp64);
FLAT_SPMV_CLEAR_INSTANTIATE(usparse::usparse_int, usparse::usparse_fp32);
FLAT_SPMV_CLEAR_INSTANTIATE(usparse::usparse_int, usparse::usparse_fpc64);
FLAT_SPMV_CLEAR_INSTANTIATE(usparse::usparse_int, usparse::usparse_fpc32);

#undef FLAT_SPMV_INSTANTIATE_MIXED
#undef FLAT_SPMV_INSTANTIATE_DEPRECATED
#undef FLAT_SPMV_ANALYSE_INSTANTIATE
#undef FLAT_SPMV_CLEAR_INSTANTIATE
