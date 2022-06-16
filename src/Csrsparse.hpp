#include <hip/hip_runtime.h>
#include "include/spgemm_dense_kernel.hpp"
#include "include/dense2csr_rocsparse.hpp"

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
    //设置BLOCKS、THREADS
    constexpr int BLOCKS=64;
    constexpr int THREADS=512;

    //将C设置为稠密格式
    // dtype *tmp_c;       //Host端结果矩阵（稠密）
    dtype *tmp_dev_C;   //Device端结果矩阵（稠密）

    // HIP_CHECK(hipHostMalloc((void**)tmp_c,m*n*sizeof(dtype)));
    HIP_CHECK(hipMalloc((void**)&tmp_dev_C,m*n*sizeof(dtype)));
    HIP_CHECK(hipMemset(tmp_dev_C,0,m*n*sizeof(dtype)));
    
    division_by_rowA<<<BLOCKS,THREADS>>>(alpha,m,n,dptr_offset_A,dptr_colindex_A,dptr_value_A,dptr_offset_B,dptr_colindex_B,dptr_value_B,tmp_dev_C);

    dense2csr(row_first,m,n,tmp_dev_C,dptr_offset_C,pdptr_colindex_C,pdptr_value_C);
 
    *ptr_nnz_C=dptr_offset_C[m];

    HIP_CHECK(hipFree(tmp_dev_C));
}

