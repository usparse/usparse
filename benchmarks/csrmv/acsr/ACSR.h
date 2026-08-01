#ifndef SPMV_ACC_BENCHMARK_ACSR_SPMV
#define SPMV_ACC_BENCHMARK_ACSR_SPMV

#include "../utils/benchmark_time.h"
#include "usparse_matrix.h"

template <typename INDEX_TYPE, typename DATA_TYPE>
void acsr(int trans, const DATA_TYPE alpha, const csr_desc<INDEX_TYPE, DATA_TYPE> h_csr_desc,
          const csr_desc<INDEX_TYPE, DATA_TYPE> d_csr_desc, const DATA_TYPE *x, DATA_TYPE *y, BenchmarkTime *bmt);

#endif
