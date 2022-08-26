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
    T* tmp_dev_value_c; 
    if(((double)nnz_B/(double)(k*n))>0.05){
        if(m>20000){
            if(buffer_data == nullptr) {
                const size_t buff2 = (nnz_calc<T,I,256, 4>)(m, n, nullptr, nullptr);
                return m*n*sizeof(T) + nnz_A*sizeof(I) + buff2;
            }
            //设置BLOCKS、THREADS
            const int BLOCKS=m;
            constexpr int THREADS=512;
            constexpr int CHUNK_SIZE=2048;
            constexpr int VEC_SIZE=16;

            I *workspace_b = (I *)buffer_data;
            tmp_dev_value_c = (T *) (workspace_b+nnz_A);   //Device端结果矩阵（稠密）

            wf_per_nnzA_kernel<T,I,CHUNK_SIZE,THREADS,VEC_SIZE><<<BLOCKS,THREADS>>>(alpha, m, n, n, dptr_offset_A, dptr_colindex_A, dptr_value_A, dptr_offset_B, dptr_colindex_B, dptr_value_B, tmp_dev_value_c,workspace_b);

        }else{
            if(buffer_data == nullptr) {
                const size_t buff2 = (nnz_calc<T,I,256, 4>)(m, n, nullptr, nullptr);
                return m*n*sizeof(T)  + buff2;
            }
            //设置BLOCKS、THREADS
            const int BLOCKS=m;
            constexpr int THREADS=1024;
            constexpr int CHUNK_SIZE=4096;

            tmp_dev_value_c = (T *) (buffer_data);   //Device端结果矩阵（稠密）

            wf_per_nnzA_kernel<T,I,CHUNK_SIZE,THREADS><<<BLOCKS,THREADS>>>(alpha, m, n, n, dptr_offset_A, dptr_colindex_A, dptr_value_A, dptr_offset_B, dptr_colindex_B, dptr_value_B, tmp_dev_value_c);
            
        }
        
        HIP_CHECK(hipMemset(dptr_offset_C,0,(m+1)*sizeof(I)));

        I *buff2 = (I *)(tmp_dev_value_c + m*n);
        dense2csr(m,n,tmp_dev_value_c,dptr_offset_C,pdptr_colindex_C,pdptr_value_C,ptr_nnz_C, buff2);
    }else{
        if(buffer_data == nullptr) {
            const size_t buff2 = (nnz_calc<T,I,256, 4>)(m, n, nullptr, nullptr);
            return m*n*sizeof(T) + m*n*sizeof(I) + nnz_A*sizeof(I) + buff2 ;
        }
        //设置BLOCKS、THREADS
        const int BLOCKS=m;
        constexpr int THREADS=512;
        constexpr int CHUNK_SIZE=2048;
        constexpr int VEC_SIZE=16;

        I *workspace_b = (I *)buffer_data;
        I *tmp_dev_col_c = (I *)(workspace_b+nnz_A);    //稠密col矩阵
        tmp_dev_value_c = (T *) (tmp_dev_col_c+m*n);   //稠密value矩阵

        HIP_CHECK(hipMemset(dptr_offset_C,0,(m+1)*sizeof(I)));

        wf_per_nnzA_kernel<T,I,CHUNK_SIZE,THREADS,VEC_SIZE><<<BLOCKS,THREADS>>>(alpha, m, n, n, dptr_offset_A, dptr_colindex_A, dptr_value_A, dptr_offset_B, dptr_colindex_B, dptr_value_B, dptr_offset_C,tmp_dev_col_c,tmp_dev_value_c,workspace_b);

        I *buff2 = (I *)(tmp_dev_value_c + m*n);
        dense2csr(m,n,tmp_dev_col_c,tmp_dev_value_c,dptr_offset_C,pdptr_colindex_C,pdptr_value_C,ptr_nnz_C, buff2);
    }
    return 0;
};

#endif // SPGEMM_DENSE_IMPL_HPP
