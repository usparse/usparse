//
// Created by genshen on 2026/08/02.
//

#ifndef USPARSE_CSRMV_BENCHMARK_MUSPARSE_HPP
#define USPARSE_CSRMV_BENCHMARK_MUSPARSE_HPP

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <musparse.h>

#include <type_traits>
#include <usparse_matrix.h>

#include "../utils/benchmark_time.h"
#include "timer/timer.h"

#define MUSPARSE_SAFE_CALL(call)                                                                                       \
  {                                                                                                                    \
    musparseStatus_t status = call;                                                                                    \
    if (status != MUSPARSE_STATUS_SUCCESS) {                                                                           \
      fprintf(stderr, "muSPARSE error at %s:%d - code %d\n", __FILE__, __LINE__, status);                              \
      exit(-1);                                                                                                        \
    }                                                                                                                  \
  }

template <typename I, typename T> struct MuSparse : CsrSpMV<I, T> {
  void csr_spmv_impl(int trans, const T alpha, const T beta, const csr_desc<I, T> h_csr_desc,
                     const csr_desc<I, T> d_csr_desc, const T *x, T *y, BenchmarkTime *bmt) {
    const T hip_alpha = static_cast<T>(alpha);
    const T hip_beta = static_cast<T>(beta);

    musparseIndexType_t idxType = (sizeof(I) == 4) ? MUSPARSE_INDEX_32I : MUSPARSE_INDEX_64I;
    musparseDataType_t dataType;
    if constexpr (std::is_same<T, float>::value) {
      dataType = MUSA_R_32F;
    } else if constexpr (std::is_same<T, double>::value) {
      dataType = MUSA_R_64F;
    } else {
      static_assert(sizeof(T) == 0, "Unsupported data type");
    }

    hip::timer::event_timer pre_timer, calc_timer, destroy_timer;

    pre_timer.start();
    musparseHandle_t handle;
    MUSPARSE_SAFE_CALL(musparseCreate(&handle));
    musparseSpMatDescr_t matA;
    MUSPARSE_SAFE_CALL(musparseCreateCsr(&matA, static_cast<int64_t>(d_csr_desc.rows),
                                         static_cast<int64_t>(d_csr_desc.cols), static_cast<int64_t>(d_csr_desc.nnz),
                                         const_cast<I *>(d_csr_desc.row_ptr),   // 去掉 const
                                         const_cast<I *>(d_csr_desc.col_index), // 去掉 const
                                         const_cast<T *>(d_csr_desc.values),    // 去掉 const
                                         idxType, idxType, MUSPARSE_INDEX_BASE_ZERO, dataType));

    musparseDnVecDescr_t vecX, vecY;
    MUSPARSE_SAFE_CALL(musparseCreateDnVec(&vecX, static_cast<int64_t>(d_csr_desc.cols), const_cast<T *>(x),
                                           dataType)); // x 是 const 指针
    MUSPARSE_SAFE_CALL(musparseCreateDnVec(&vecY, static_cast<int64_t>(d_csr_desc.rows), y, dataType));

    musparseOperation_t op = (trans == 0) ? MUSPARSE_OPERATION_NON_TRANSPOSE : MUSPARSE_OPERATION_TRANSPOSE;

    size_t bufferSize = 0;
    MUSPARSE_SAFE_CALL(musparseSpMV_bufferSize(handle, op, &alpha, matA, vecX, &beta, vecY, dataType,
                                               MUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));

    void *dBuffer = nullptr;
    if (bufferSize > 0) {
      musaError_t err = musaMalloc(&dBuffer, bufferSize);
      if (err != musaSuccess) {
        fprintf(stderr, "musaMalloc failed: %d\n", err);
        exit(-1);
      }
    }
    pre_timer.stop();

    calc_timer.start();
    MUSPARSE_SAFE_CALL(
        musparseSpMV(handle, op, &alpha, matA, vecX, &beta, vecY, dataType, MUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
    calc_timer.stop(true);

    destroy_timer.start();
    if (dBuffer)
      musaFree(dBuffer);
    MUSPARSE_SAFE_CALL(musparseDestroyDnVec(vecX));
    MUSPARSE_SAFE_CALL(musparseDestroyDnVec(vecY));
    MUSPARSE_SAFE_CALL(musparseDestroySpMat(matA));
    MUSPARSE_SAFE_CALL(musparseDestroy(handle));
    destroy_timer.stop();

    if (bmt != nullptr) {
      bmt->set_time(pre_timer.time_use, calc_timer.time_use, 0.0, destroy_timer.time_use);
    }
  }
  bool verify_beta_y() { return true; }
};

#endif // USPARSE_CSRMV_BENCHMARK_MUSPARSE_HPP
