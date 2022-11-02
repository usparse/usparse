#include <hip/hip_runtime.h>
#include "api/spgemm_hip_api.h"
#include "api/spgemm_hip_types.h"


hipError_t hipFree(double* ptr) {
	return hipSuccess;
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
   sparse_gemm_hip<dtype, csrIdxType>(hip_operation_none, hip_operation_none, alpha, 0, m, n, k,nnz_A,dptr_offset_A,dptr_colindex_A,dptr_value_A,nnz_B,dptr_offset_B,dptr_colindex_B,dptr_value_B,ptr_nnz_C,dptr_offset_C,pdptr_colindex_C,pdptr_value_C);
}
