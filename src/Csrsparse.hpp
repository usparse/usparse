#include <hip/hip_runtime.h>
#include "include/spgemm_dense_kernel.hpp"
#include "include/dense2csr.hpp"

// We offer a parameter `buffer_data`.
// If it is nullprt, deveice memory will allocated.
size_t call_device_spgemm_imp(const int transA,
		const int          transB,
		const dtype        alpha,
		const size_t       m,
		const size_t       n,
		const size_t       k,
        const size_t       nnz_A,
		const csrIdxType*  dptr_offset_A,
		const csrIdxType*  dptr_colindex_A,
		const dtype*       dptr_value_A,
        const size_t       nnz_B,
		const csrIdxType*  dptr_offset_B,
		const csrIdxType*  dptr_colindex_B,
		const dtype*       dptr_value_B,
        size_t*            ptr_nnz_C,
        csrIdxType*        dptr_offset_C,
        csrIdxType**       pdptr_colindex_C,
		dtype**            pdptr_value_C,
        void *buffer_data) {
    if(buffer_data == nullptr) {
        const size_t buff2 = (nnz_calc<64, 256, 4>)(m, n, nullptr, n, nullptr, nullptr);
        // dense array + buff2
        return m*n*sizeof(dtype) + buff2;
    }
    //设置BLOCKS、THREADS
    constexpr int BLOCKS=128;
    constexpr int THREADS=1024;

    //将C设置为稠密格式
    // dtype *tmp_c;       //Host端结果矩阵（稠密）
    dtype *tmp_dev_C = (dtype *) buffer_data;   //Device端结果矩阵（稠密）

    // division_by_row_thread_per_row<<<BLOCKS,THREADS>>>(alpha,m,n,dptr_offset_A,dptr_colindex_A,dptr_value_A,dptr_offset_B,dptr_colindex_B,dptr_value_B,tmp_dev_C);
    // division_by_row_thread_per_nnz<<<BLOCKS,THREADS,(m/BLOCKS+1)*sizeof(int)>>>(alpha, m, n,dptr_offset_A,dptr_colindex_A,dptr_value_A,dptr_offset_B,dptr_colindex_B,dptr_value_B,tmp_dev_C);
    division_by_row_wf_per_nnz<THREADS,64,3><<<BLOCKS,THREADS,(m/BLOCKS+1)*sizeof(int)>>>(alpha, m, n,n,dptr_offset_A,dptr_colindex_A,dptr_value_A,dptr_offset_B,dptr_colindex_B,dptr_value_B,tmp_dev_C);
    HIP_CHECK(hipMemset(dptr_offset_C,0,(m+1)*sizeof(csrIdxType)));
    int *buff2 = (int *)(tmp_dev_C + m*n);
    dense2csr(m,n,tmp_dev_C,dptr_offset_C,pdptr_colindex_C,pdptr_value_C,ptr_nnz_C, buff2);
    return 0;
}

void  call_device_spgemm(const int transA,
		const int          transB,
		const dtype        alpha,
		const size_t       m,
		const size_t       n,
		const size_t       k,
        const size_t       nnz_A,
		const csrIdxType*  dptr_offset_A,
		const csrIdxType*  dptr_colindex_A,
		const dtype*       dptr_value_A,
        const size_t       nnz_B,
		const csrIdxType*  dptr_offset_B,
		const csrIdxType*  dptr_colindex_B,
		const dtype*       dptr_value_B,
        size_t*            ptr_nnz_C,
        csrIdxType*        dptr_offset_C,
        csrIdxType**       pdptr_colindex_C,
		dtype**            pdptr_value_C )
{
    const size_t buff_size = call_device_spgemm_imp(transA, transB, alpha, m,n,k,
        nnz_A, dptr_offset_A, dptr_colindex_A, dptr_value_A,
        nnz_B, dptr_offset_B, dptr_colindex_B, dptr_value_B,
        ptr_nnz_C, dptr_offset_C, pdptr_colindex_C, pdptr_value_C,
        nullptr);
    
    void *buffer_data = nullptr;
    // HIP_CHECK(hipHostMalloc((void**)tmp_c,m*n*sizeof(dtype)));
    HIP_CHECK(hipMalloc((void**)&buffer_data, buff_size));
    HIP_CHECK(hipMemset(buffer_data,0, buff_size));

    call_device_spgemm_imp(transA, transB, alpha, m,n,k,
        nnz_A, dptr_offset_A, dptr_colindex_A, dptr_value_A,
        nnz_B, dptr_offset_B, dptr_colindex_B, dptr_value_B,
        ptr_nnz_C, dptr_offset_C, pdptr_colindex_C, pdptr_value_C,
        buffer_data);

    HIP_CHECK(hipFree(buffer_data));
}
