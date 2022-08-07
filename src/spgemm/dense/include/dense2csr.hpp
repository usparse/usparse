#pragma once

#include <hip/hip_runtime.h>

#include "dense2csr_kernel.hpp"
#include "nnz_calc_kernel.hpp"

void dense2csr(int m,int n,csrValueType *dev_dense,csrIndexType *ptr_offset,csrIndexType **ptr_colindex,csrValueType **ptr_value,size_t* ptr_nnz, int *temp_mem){
    // calculate nnz per row, saved in ptr_offset
    nnz_kernel_wrapper(m, n, dev_dense, n, ptr_offset);
    // note: template paratemer should be the same as memory allocation.
    (nnz_calc<64, 256, 4>)(m,n,dev_dense,n,ptr_offset, temp_mem);
    hipDeviceSynchronize(); // wait memory storing finishing for using `ptr_offset[m]`
    const csrIndexType nnz = ptr_offset[m];
    *ptr_nnz=nnz;

    // allocate memory.
    // note: memset is not necessary (in kernel function, it does not load data from device memory).
    hipMalloc((void **)ptr_colindex,(nnz)*sizeof(csrIdxType));
    hipMalloc((void **)ptr_value,(nnz)*sizeof(dtype));

    // const size_t csr_size = nnz*sizeof(csrIdxType) + nnz*sizeof(dtype);
    // void *dptr_csr_c = nullptr;
    // hipMalloc(&dptr_csr_c, csr_size); 
    // note: memset is not necessary (in kernel function, it does not load data from device memory).
    // *ptr_colindex = (csrIndexType *)dptr_csr_c;
    // *ptr_value = (csrValueType *)(ptr_colindex + nnz);

    dense2csx_template(m,n,dev_dense,n,*ptr_value,ptr_offset,*ptr_colindex);
}
