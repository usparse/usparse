#ifndef DENSETOSPARSE_HPP
#define DENSETOSPARSE_HPP

#include <hip/hip_runtime.h>

template <unsigned int BLOCKSIZE, typename I, typename T>
__launch_bounds__(BLOCKSIZE) __global__
    void densetosparse_gthr_kernel(I nnz, const T *y, T *x_val, const I *x_ind, const int idx_base) {
  I idx = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

  if (idx >= nnz) {
    return;
  }

  x_val[idx] = y[x_ind[idx] - idx_base];
}
void densetosparse_dgthr(const int nnz, const double *y, double *x_val, const int *x_ind, const int idx_base) {
#define GTHR_DIM 512
  dim3 gthr_blocks((nnz - 1) / GTHR_DIM + 1);
  dim3 gthr_threads(GTHR_DIM);

  hipLaunchKernelGGL((densetosparse_gthr_kernel<GTHR_DIM>), gthr_blocks, gthr_threads, 0, 0, nnz, y, x_val, x_ind,
                     idx_base);
#undef GTHR_DIM
}
void densetosparse_sgthr(const int nnz, const float *y, float *x_val, const int *x_ind, const int idx_base) {
#define GTHR_DIM 512
  dim3 gthr_blocks((nnz - 1) / GTHR_DIM + 1);
  dim3 gthr_threads(GTHR_DIM);

  hipLaunchKernelGGL((densetosparse_gthr_kernel<GTHR_DIM>), gthr_blocks, gthr_threads, 0, 0, nnz, y, x_val, x_ind,
                     idx_base);
#undef GTHR_DIM
}
template <typename T>
void densetosparse_gthr_wrapper(const int nnz, const T *y, T *x_val, const int *x_ind, const int idx_base);

template <>
void densetosparse_gthr_wrapper<float>(const int nnz, const float *y, float *x_val, const int *x_ind,
                                       const int idx_base) {
  // ROCSPARSE_CHECK( rocsparse_sgthr(handle, nnz, y, x_val, x_ind, idx_base) )
  densetosparse_sgthr(nnz, y, x_val, x_ind, idx_base);
}

template <>
void densetosparse_gthr_wrapper<double>(const int nnz, const double *y, double *x_val, const int *x_ind,
                                        const int idx_base) {
  // ROCSPARSE_CHECK( rocsparse_dgthr(handle, nnz, y, x_val, x_ind, idx_base) )
  densetosparse_dgthr(nnz, y, x_val, x_ind, idx_base);
}
#endif
