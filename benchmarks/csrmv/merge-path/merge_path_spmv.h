#ifndef SPMV_ACC_BENCHMARK_MERGE_PATH_SPMV_H
#define SPMV_ACC_BENCHMARK_MERGE_PATH_SPMV_H

#include "../utils/benchmark_time.h"
#include "merge_path_config.h"
#include "usparse_matrix.h"

template <typename DATA_TYPE, int REDUCTION_ALGORITHM, int UPDATE_ALGORITHM>
void merge_path_spmv(int trans, const DATA_TYPE alpha, const DATA_TYPE beta, const csr_desc<int, DATA_TYPE> h_csr_desc,
                     const csr_desc<int, DATA_TYPE> d_csr_desc, const DATA_TYPE *x, DATA_TYPE *y, BenchmarkTime *bmt);

#endif // SPMV_ACC_BENCHMARK_MERGE_PATH_SPMV_H