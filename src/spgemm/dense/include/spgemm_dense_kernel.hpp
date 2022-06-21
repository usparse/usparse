#ifndef SPGEMM_DENSE_SPGEMM_DENSE_KERNEL_H
#define SPGEMM_DENSE_SPGEMM_DENSE_KERNEL_H
#include "types.hpp"

//每个THREAD计算A的一行
__global__ void division_by_row_thread_per_row(const csrValueType alpha,const size_t dev_A_row,const size_t dev_B_col,const csrIndexType* dptr_offset_A,
const csrIndexType* dptr_colindex_A,const csrValueType* dptr_value_A,const csrIndexType* dptr_offset_B,
const csrIndexType* dptr_colindex_B,const csrValueType* dptr_value_B,csrValueType* tmp_dev_c);


//每个BLOCK计算A的若干行，每个thread共同处理一行
__global__ void division_by_row_thread_per_nnz(const xtype alpha,const csrIndexType dev_A_row,const csrIndexType dev_B_col,
const csrIndexType *dptr_offset_A,const csrIndexType *dptr_colindex_A,
const csrValueType *dptr_value_A,const csrIndexType *dptr_offset_B,const csrIndexType *dptr_colindex_B,
const csrValueType *dptr_value_B,xtype *tmp_dev_c);


#endif //SPGEMM_DENSE_SPGEMM_DENSE_KERNEL_H