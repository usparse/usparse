#ifndef SPGEMM_DENSE_IMPL_HPP
#define SPGEMM_DENSE_IMPL_HPP

#include "dense_kernel.hpp"
#include "dense2csr_impl.hpp"
#include "common.hpp"

// We offer a parameter `buffer_data`.
// If it is nullprt, deveice memory will allocated.
template <typename T, typename I>
size_t dense_spgemm(const hip_sparse_operation transA,
		const hip_sparse_operation          transB,
		const T            alpha,
		const size_t       m,
		const size_t       n,
		const size_t       k,
        const size_t       nnz_A,
		const I*           dptr_offset_A,
		const I*           dptr_colindex_A,
		const T*           dptr_value_A,
        const size_t       nnz_B,
		const I*           dptr_offset_B,
		const I*           dptr_colindex_B,
		const T*           dptr_value_B,
        size_t*            ptr_nnz_C,
        I*                 dptr_offset_C,
        I**                pdptr_colindex_C,
		T**                pdptr_value_C,
        void *buffer_data) {
    if(buffer_data == nullptr) {
        const size_t buff2 = (nnz_calc<T,I,256, 4>)(m, n, nullptr, n, nullptr, nullptr);
        // dense array + buff2
        return m*n*sizeof(T) + buff2;
    }
    //设置BLOCKS、THREADS
    constexpr int BLOCKS=64;
    constexpr int THREADS=1024;

    T *tmp_dev_C = (T *) buffer_data;   //Device端结果矩阵（稠密）

    // division_by_row_thread_per_row<<<BLOCKS,THREADS>>>(alpha,m,n,dptr_offset_A,dptr_colindex_A,dptr_value_A,dptr_offset_B,dptr_colindex_B,dptr_value_B,tmp_dev_C);
    // division_by_row_thread_per_nnz<<<BLOCKS,THREADS,(m/BLOCKS+1)*sizeof(int)>>>(alpha, m, n,dptr_offset_A,dptr_colindex_A,dptr_value_A,dptr_offset_B,dptr_colindex_B,dptr_value_B,tmp_dev_C);
    wf_per_nnzA_kernel<T,I,THREADS,4><<<BLOCKS,THREADS,(m/BLOCKS+1)*sizeof(int)>>>(alpha, m, n,n,dptr_offset_A,dptr_colindex_A,dptr_value_A,dptr_offset_B,dptr_colindex_B,dptr_value_B,tmp_dev_C);
    HIP_CHECK(hipMemset(dptr_offset_C,0,(m+1)*sizeof(I)));
    I *buff2 = (I *)(tmp_dev_C + m*n);
    dense2csr(m,n,tmp_dev_C,dptr_offset_C,pdptr_colindex_C,pdptr_value_C,ptr_nnz_C, buff2);
    return 0;
};

#endif // SPGEMM_DENSE_IMPL_HPP
