#ifndef SPGEMM_DENSE2CSR_IMPL_HPP
#define SPGEMM_DENSE2CSR_IMPL_HPP

#include <hip/hip_runtime.h>

#include "dense2csr_kernel.hpp"
#include "nnz_impl.hpp"
#include "building_config.h"

template <typename T, typename I>
void dense2csr_template(const size_t            m,
                        const size_t            n,
                        const T*                A,
                        const size_t            ld,
                        T*                      csx_val,
                        I*                      csx_row_ptr,
                        I*                      csx_col_ind)
{
    static constexpr int data_ratio = sizeof(T) / sizeof(float);
    static constexpr int NROWS_PER_BLOCK = 16 / (data_ratio > 0 ? data_ratio : 1);
    int                  blocks          = (m - 1) / NROWS_PER_BLOCK + 1;
    dim3                 k_blocks(blocks), k_threads(__WF_SIZE__ * NROWS_PER_BLOCK);
    dense2csr_kernel<T, I, NROWS_PER_BLOCK><<<k_blocks,k_threads>>>(m,n,A,ld,csx_val,csx_row_ptr,csx_col_ind);
}

template <typename T, typename I>
void dense2csr(const size_t m,const size_t n,T *dev_dense,I *ptr_offset,I **ptr_colindex,T **ptr_value,size_t* ptr_nnz, I *temp_mem){
    // calculate nnz per row, saved in ptr_offset
    nnz_kernel_wrapper(m, n, dev_dense, n, ptr_offset);
    // note: template paratemer should be the same as memory allocation.
    (nnz_calc< T,I,256, 4>)(m,n,dev_dense,n,ptr_offset, temp_mem);
    hipDeviceSynchronize(); // wait memory storing finishing for using `ptr_offset[m]`
    const size_t nnz = ptr_offset[m];
    *ptr_nnz=nnz;

    // allocate memory.
    // note: memset is not necessary (in kernel function, it does not load data from device memory).
    hipMalloc((void **)ptr_colindex,(nnz)*sizeof(I));
    hipMalloc((void **)ptr_value,(nnz)*sizeof(T));

    dense2csr_template(m,n,dev_dense,n,*ptr_value,ptr_offset,*ptr_colindex);
}

#endif // SPGEMM_DENSE2CSR_IMPL_HPP
