//
// Created by luxu on 2022/12/08.
//

#ifndef SPGEMM_BENCHMARK_ROCSPARSE_CSR2CSC_HPP
#define SPGEMM_BENCHMARK_ROCSPARSE_CSR2CSC_HPP

#include <hipsparse.h>
#include <rocsparse.h>
#include "common.h"

//matrix turn by rocsparse csr2csc
template <typename I, typename T>
void rosparse_csr_matirx_turn(csr_mtx<I, T> &csr_matrix, I *&row_ptr, I *&col_idx, T *&val){
    
    rocsparse_handle handle = NULL;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    I *csr_row_ptr,*csr_col_idx;
    T *csr_val;

    HIP_CHECK(hipMalloc((void **)&csr_row_ptr, (csr_matrix.rows + 1) * sizeof(I)))
    HIP_CHECK(hipMalloc((void **)&csr_col_idx, csr_matrix.nnz * sizeof(I)))
    HIP_CHECK(hipMalloc((void **)&csr_val, csr_matrix.nnz * sizeof(T)))
    HIP_CHECK(hipMemcpy(csr_row_ptr, csr_matrix.row_ptr, (csr_matrix.rows + 1) * sizeof(I), hipMemcpyHostToDevice))
    HIP_CHECK(hipMemcpy(csr_col_idx, csr_matrix.col_index, csr_matrix.nnz * sizeof(I), hipMemcpyHostToDevice))
    HIP_CHECK(hipMemcpy(csr_val, csr_matrix.values, csr_matrix.nnz * sizeof(T), hipMemcpyHostToDevice))

    HIP_CHECK(hipMalloc((void **)&row_ptr, (csr_matrix.cols + 1) * sizeof(I)))
    HIP_CHECK(hipMalloc((void **)&col_idx, csr_matrix.nnz * sizeof(I)))
    HIP_CHECK(hipMalloc((void **)&val, csr_matrix.nnz * sizeof(T)))

    size_t buffer_size;
    ROCSPARSE_CHECK(rocsparse_csr2csc_buffer_size(handle,csr_matrix.rows,csr_matrix.cols,csr_matrix.nnz,csr_row_ptr,csr_col_idx,rocsparse_action_numeric,&buffer_size))

    void *temp_buffer;
    HIP_CHECK(hipMalloc(&temp_buffer, buffer_size))
    ROCSPARSE_CHECK(rocsparse_dcsr2csc(handle,csr_matrix.rows,csr_matrix.cols,csr_matrix.nnz,csr_val,csr_row_ptr,csr_col_idx, val, col_idx, row_ptr,rocsparse_action_numeric,rocsparse_index_base_zero,temp_buffer))

    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle))

    HIP_CHECK(hipFree(temp_buffer))
    HIP_CHECK(hipFree(csr_row_ptr))
    HIP_CHECK(hipFree(csr_col_idx))
    HIP_CHECK(hipFree(csr_val))

}

#endif // SPGEMM_BENCHMARK_ROCSPARSE_CSR2CSC_HPP
