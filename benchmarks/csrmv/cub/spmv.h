#ifndef SPMV_ACC_BENCHMARK_CUB_SPMV
#define SPMV_ACC_BENCHMARK_CUB_SPMV

#include "../utils/benchmark_time.h"
#include "usparse_matrix.h"

template <typename DATA_TYPE>
void spmv(int trans, const csr_desc<int, DATA_TYPE> h_csr_desc, const csr_desc<int, DATA_TYPE> d_csr_desc,
          const DATA_TYPE *x, DATA_TYPE *y, BenchmarkTime *bmt);

#endif