#include <hip/hip_runtime.h>

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
    // Get the  nonzero num of C and the value in offset_C 
    size_t nonzero = max(nnz_A, nnz_B);
    *ptr_nnz_C = nonzero;

    // Malloc pdptr_colindex_C and pdptr_value_C 
    HIP_CHECK( hipMalloc((void**) pdptr_colindex_C, nonzero * sizeof(csrIdxType)) )
    HIP_CHECK( hipMalloc((void**) pdptr_value_C, nonzero * sizeof(dtype)) )

    // Kernels need to implement 
    // ...
}

