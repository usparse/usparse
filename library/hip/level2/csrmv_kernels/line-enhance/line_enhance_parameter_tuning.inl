//
// Created by genshen on 2026/1/18.
//

#ifndef USPARSE_LINE_ENHANCE_PARAMETER_TUNING_H
#define USPARSE_LINE_ENHANCE_PARAMETER_TUNING_H

template <typename T, typename NNZ, typename ROW, typename COL, typename SCALE, typename A, typename X, typename Y,
          int WF_SIZE>
void adaptive_enhance_sparse_spmv_gfx936(hipStream_t stream, int trans, const SCALE alpha, const SCALE beta,
                                         const usparse::csr_mat_descr<A, NNZ, ROW, COL> d_csr_desc, const X *x, Y *y) {
  EXPAND_FROM_CSR_MAT_DESCR(A, NNZ, ROW, COL, d_csr_desc)
  if (m > 0) {
    double density = static_cast<double>(d_csr_desc.nnz) / m;

#define LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(_R, _ROWS, _VEC, _THREADS)                                 \
  {                                                                                                                    \
    constexpr int _REDUCE_OPT = (_VEC <= 1) ? LE_REDUCE_OPTION_DIRECT : LE_REDUCE_OPTION_VEC;                          \
    int _BLOCKS = m / _ROWS + (m % _ROWS == 0 ? 0 : 1);                                                                \
    static_assert(_ROWS * _VEC <= _THREADS, "ROWS_PER_BLOCK * VEC_SIZE must be <= THREADS_PER_BLOCK");                 \
    LINE_ENHANCE_KERNEL_WRAPPER(stream, T, NNZ, ROW, A, X, Y, _REDUCE_OPT, _ROWS, WF_SIZE, _VEC, _R, _BLOCKS,          \
                                _THREADS);                                                                             \
    return;                                                                                                            \
  }

    if (m < 1000) { // Group 0: Tiny
      if (density < 2) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(1, 128, 1, 512);
      } else if (density < 4) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(1, 128, 2, 512);
      } else if (density < 8) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(2, 128, 2, 512);
      } else if (density < 16) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(2, 64, 8, 512);
      } else if (density < 32) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(4, 64, 8, 512);
      } else if (density < 64) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(4, 32, 16, 512);
      } else {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(4, 16, 32, 512);
      }
    } else if (m < 10000) { // Group 1: Small
      if (density < 2) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(1, 128, 2, 256);
      } else if (density < 4) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(3, 128, 4, 512);
      } else if (density < 8) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(3, 128, 4, 512);
      } else if (density < 16) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(4, 128, 4, 512);
      } else if (density < 32) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(4, 128, 4, 512);
      } else if (density < 64) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(4, 16, 16, 512);
      } else {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(4, 16, 32, 512);
      }
    } else if (m < 50000) { // Group 2: LowMed
      if (density < 2) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(1, 128, 1, 256);
      } else if (density < 4) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(1, 128, 2, 512);
      } else if (density < 8) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(4, 16, 4, 256);
      } else if (density < 16) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(4, 16, 32, 512);
      } else if (density < 32) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(4, 16, 16, 512);
      } else if (density < 64) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(4, 16, 32, 512);
      } else {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(4, 16, 32, 512);
      }
    } else if (m < 200000) { // Group 3: Medium
      if (density < 2) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(2, 64, 4, 256);
      } else if (density < 4) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(3, 32, 16, 512);
      } else if (density < 8) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(4, 128, 2, 256);
      } else if (density < 16) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(4, 16, 32, 512);
      } else if (density < 32) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(4, 128, 4, 512);
      } else if (density < 64) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(4, 32, 8, 256);
      } else {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(4, 16, 32, 512);
      }
    } else if (m < 1000000) { // Group 4: HighMed
      if (density < 2) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(2, 32, 8, 256);
      } else if (density < 4) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(2, 128, 1, 256);
      } else if (density < 8) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(4, 128, 2, 256);
      } else if (density < 16) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(4, 128, 2, 256);
      } else if (density < 32) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(2, 32, 2, 256);
      } else if (density < 64) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(4, 64, 8, 512);
      } else {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(3, 16, 8, 256);
      }
    } else if (m < 10000000) { // Group 5: Large
      if (density < 2) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(3, 16, 8, 256);
      } else if (density < 4) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(2, 128, 1, 256);
      } else if (density < 8) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(2, 128, 1, 512);
      } else if (density < 16) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(2, 32, 8, 256);
      } else if (density < 32) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(3, 64, 2, 256);
      } else if (density < 64) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(3, 16, 8, 256);
      } else {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(3, 32, 8, 512);
      }
    } else { // Group 6: Huge (>= 10M)
      if (density < 2) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(3, 32, 8, 512);
      } else if (density < 4) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(2, 128, 1, 256);
      } else if (density < 8) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(1, 128, 2, 256);
      } else if (density < 16) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(1, 128, 2, 256);
      } else if (density < 32) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(3, 32, 8, 256);
      } else if (density < 64) {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(3, 32, 8, 256);
      } else {
        LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936(3, 32, 8, 256);
      }
    }

#undef LAUNCH_LINE_ENHANCE_WITH_COMPILE_TIME_TUNING_GFX936
  } else {  // m == 0
    return; // invalid. do nothing.
  }
  return;
}

#endif // USPARSE_LINE_ENHANCE_PARAMETER_TUNING_H
