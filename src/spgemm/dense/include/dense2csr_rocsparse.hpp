#pragma once

#include "types.hpp"

//row_first:稠密矩阵行优先存储:调用dense2csc接口,只支持方阵
//col_first:稠密矩阵列优先存储:调用dense2csr接口,支持任意矩阵
void dense2csr(dense_matrix_storage_option dense_matrix_storage_option,const size_t dense_row,const size_t dense_col,const csrValueType *dev_dense,csrIndexType *ptr_offset,csrIndexType **ptr_colindex,
csrValueType **ptr_value){
    csrIndexType* nnz_per_row_columns;
    csrIndexType* nnz_total_dev_host_ptr;
    HIP_CHECK(hipMalloc((void**)&nnz_per_row_columns,dense_row*sizeof(csrIndexType))); 
    HIP_CHECK(hipMalloc((void**)&nnz_total_dev_host_ptr,sizeof(csrIndexType))); 
    
    rocsparse_mat_descr descr; 
    rocsparse_create_mat_descr(&descr);

    rocsparse_handle handle = NULL;
    rocsparse_create_handle(&handle);

    if(dense_matrix_storage_option==1){
        rocsparse_dnnz(handle,rocsparse_direction_column,dense_row, dense_col,descr,dev_dense,dense_row,nnz_per_row_columns,nnz_total_dev_host_ptr);

        HIP_CHECK(hipMalloc((void**)ptr_colindex,(*nnz_total_dev_host_ptr)*sizeof(csrIndexType)));
        HIP_CHECK(hipMalloc((void**)ptr_value,(*nnz_total_dev_host_ptr)*sizeof(csrValueType)));
        
        rocsparse_ddense2csc(handle, dense_row, dense_col, descr, 
                                    dev_dense, dense_row, nnz_per_row_columns,*ptr_value,
                                    ptr_offset, *ptr_colindex);
    }else{
        rocsparse_dnnz(handle,rocsparse_direction_row,dense_row, dense_col,descr,dev_dense,dense_row,nnz_per_row_columns,nnz_total_dev_host_ptr);

        HIP_CHECK(hipMalloc((void**)ptr_colindex,(*nnz_total_dev_host_ptr)*sizeof(csrIndexType)));
        HIP_CHECK(hipMalloc((void**)ptr_value,(*nnz_total_dev_host_ptr)*sizeof(csrValueType)));
        
        rocsparse_ddense2csr(handle, dense_row, dense_col, descr, 
                                    dev_dense, dense_row, nnz_per_row_columns,*ptr_value,
                                    ptr_offset, *ptr_colindex);
    }
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(nnz_per_row_columns));
    HIP_CHECK(hipFree(nnz_total_dev_host_ptr));
}