#include "include/spgemm_dense_kernel.hpp"

__global__ void division_by_rowA(const xtype alpha,const size_t dev_A_row,const size_t dev_B_col,const csrIndexType* dptr_offset_A,
const csrIndexType* dptr_colindex_A,const csrValueType* dptr_value_A,const csrIndexType* dptr_offset_B,
const csrIndexType* dptr_colindex_B,const csrValueType* dptr_value_B,csrValueType* tmp_dev_c){
    
    //每个thread计算A的一行
    int threads=gridDim.x*blockDim.x;
    int tid = threadIdx.x+blockIdx.x*blockDim.x;

    for(int i=tid;i<dev_A_row;i=i+threads){
        //获取当前处理的行(变量i)
        int row_A_start=dptr_offset_A[i];
        int row_A_end=dptr_offset_A[i+1];
        for (int j=row_A_start;j<row_A_end;j++){
            int A_col=dptr_colindex_A[j];
            int row_B_start=dptr_offset_B[A_col];
            int row_B_end=dptr_offset_B[A_col+1];
            for(int k=row_B_start;k<row_B_end;k++){
                int C_col=dptr_colindex_B[k];
                *(tmp_dev_c+i*dev_B_col+C_col)+=alpha*(*(dptr_value_A+j))*(*(dptr_value_B+k));
            }
        }
    }
}