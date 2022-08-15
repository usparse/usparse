#pragma once

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
        return m*n*sizeof(T) + buff2;
    }
    //设置BLOCKS、THREADS
    const int BLOCKS=m;
    constexpr int THREADS=512;
    constexpr int SHARED_BUFF_NUM=4; //Chunk=SHARED_BUFF_NUM*THREADS
    constexpr int VEC_SIZE=16;

    T *tmp_dev_C = (T *) buffer_data;   //Device端结果矩阵（稠密）
    I *workspace_b = nullptr;
    hipMalloc((void **)&workspace_b, sizeof(I) * nnz_A);
    // wf_per_nnzA_kernel<T,I,THREADS,SHARED_BUFF_NUM><<<BLOCKS,THREADS,(m/BLOCKS+1)*sizeof(int)>>>(alpha, m, n,n,dptr_offset_A,dptr_colindex_A,dptr_value_A,dptr_offset_B,dptr_colindex_B,dptr_value_B,tmp_dev_C);
    // wf_per_nnzA_kernel2<T,I,THREADS,SHARED_BUFF_NUM><<<BLOCKS,THREADS>>>(alpha, m, n,n,dptr_offset_A,dptr_colindex_A,dptr_value_A,dptr_offset_B,dptr_colindex_B,dptr_value_B,tmp_dev_C);
    wf_per_nnzA_kernel3<T,I,THREADS,VEC_SIZE,SHARED_BUFF_NUM><<<BLOCKS,THREADS>>>(alpha, m, n, n, dptr_offset_A, dptr_colindex_A, dptr_value_A, dptr_offset_B, dptr_colindex_B, dptr_value_B, tmp_dev_C,workspace_b);
    HIP_CHECK(hipMemset(dptr_offset_C,0,(m+1)*sizeof(I)));
    I *buff2 = (I *)(tmp_dev_C + m*n);
    dense2csr(m,n,tmp_dev_C,dptr_offset_C,pdptr_colindex_C,pdptr_value_C,ptr_nnz_C, buff2);
    return 0;
};
