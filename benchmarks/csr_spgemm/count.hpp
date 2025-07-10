#ifndef SPGEMM_BENCHMARK_COUNT_HPP
#define SPGEMM_BENCHMARK_COUNT_HPP

template <typename I>
int symbolic_num_of_operations(const size_t n, const size_t nnz_A, const I *pColIndexA, const I *pOffsetB) {
    int count = 0;
    int per_col_num_A[n];
    int per_row_num_B[n];
    for(int i = 0; i < n; i++) {
        per_col_num_A[i] = 0;
        per_row_num_B[i] = 0; 
    }

    for(int i = 0; i < nnz_A; i++) {
        per_col_num_A[ pColIndexA[i] ] ++ ;
    }
    for(int j = 0; j < n; j++) {
        per_row_num_B[j] = pOffsetB[j + 1] - pOffsetB[j];
    }
    for(int i = 0; i < n; i ++) {
        count += per_col_num_A[i] * per_row_num_B[i];
    }
    return count;
}

template <typename I>
int numeric_nums_of_add(const size_t n, const size_t nnz_A, const I *pColIndexA, const I *pOffsetB) {
    return symbolic_num_of_operations<I>(n, nnz_A, pColIndexA, pOffsetB);    
}

template <typename I>
int numeric_nums_of_mul(const size_t n, const size_t nnz_A, const I *pColIndexA, const I *pOffsetB, const size_t nnz_C) {
    return symbolic_num_of_operations<I>(n, nnz_A, pColIndexA, pOffsetB) + nnz_C;
}

#endif // SPGEMM_BENCHMARK_COUNT_HPP