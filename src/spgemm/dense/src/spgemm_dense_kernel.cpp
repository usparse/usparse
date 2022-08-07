#include "include/spgemm_dense_kernel.hpp"

__global__ void division_by_row_thread_per_row(const xtype alpha,const size_t dev_A_row,const size_t dev_B_col,const csrIndexType* dptr_offset_A,
const csrIndexType* dptr_colindex_A,const csrValueType* dptr_value_A,const csrIndexType* dptr_offset_B,
const csrIndexType* dptr_colindex_B,const csrValueType* dptr_value_B,xtype* tmp_dev_c){
    
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

//每个BLOCK计算A的若干行，每个thread共同处理一行
__global__ void division_by_row_thread_per_nnz(const xtype alpha,const csrIndexType dev_A_row,const csrIndexType dev_B_col,
const csrIndexType *dptr_offset_A,const csrIndexType *dptr_colindex_A,
const csrValueType *dptr_value_A,const csrIndexType *dptr_offset_B,const csrIndexType *dptr_colindex_B,
const csrValueType *dptr_value_B,xtype *tmp_dev_c){
    int rowA_per_Block=dev_A_row/gridDim.x;
    int reality_rowA_per_Block=rowA_per_Block;

    if(blockIdx.x<dev_A_row%gridDim.x){
        reality_rowA_per_Block++;
    }

    int block_row_start= blockIdx.x<dev_A_row%gridDim.x ? blockIdx.x*reality_rowA_per_Block : (dev_A_row%gridDim.x)*(reality_rowA_per_Block+1)+(blockIdx.x-dev_A_row%gridDim.x)*reality_rowA_per_Block;
    int block_row_end= blockIdx.x<dev_A_row%gridDim.x ? (blockIdx.x+1)*reality_rowA_per_Block : (dev_A_row%gridDim.x)*(reality_rowA_per_Block+1)+(blockIdx.x-dev_A_row%gridDim.x+1)*reality_rowA_per_Block;

    // 计算每个block需要计算的nnz数量
    int block_nnz=dptr_offset_A[block_row_end]-dptr_offset_A[block_row_start];
    
    // 动态申请LDS
    // extern __shared__ int s[];
    // int *s_offset_A=(int*)s;
    // int *s_offset_B=(int *)&s_offset_A[rowA_per_Block+1];
    extern __shared__ int s_offset_A[];
    
    for(int i=threadIdx.x;i<reality_rowA_per_Block+1;i=i+blockDim.x){
        s_offset_A[i]=dptr_offset_A[block_row_start+i];
    }

    __syncthreads();

    int s_A_row=0;//记录在LDS中的行偏移
    for(int i=threadIdx.x;i<block_nnz;i=i+blockDim.x){
        int A_col_id=s_offset_A[0]+i;
        int A_col=dptr_colindex_A[A_col_id];
        //确定当前元素A的行
        for(int j=s_A_row;j<reality_rowA_per_Block+1;j++){
            if (A_col_id>=s_offset_A[s_A_row]&&A_col_id<s_offset_A[s_A_row+1]){
                break;
            }else{
                s_A_row++;
            }
        }
        int A_row=s_A_row+block_row_start;
        //获取B的行偏移及列索引并完成计算
        int B_row_start=dptr_offset_B[A_col];
        int B_row_end=dptr_offset_B[A_col+1];
        for(int j=B_row_start;j<B_row_end;j++){
            int C_col=dptr_colindex_B[j];
            atomicAdd(tmp_dev_c+A_row*dev_B_col+C_col,alpha*(dptr_value_A[A_col_id])*(dptr_value_B[j]));
        }
    }

    // 不使用LDS的代码    
    // int A_row=block_row_start;
    // for(int i=threadIdx.x;i<block_nnz;i=i+blockDim.x){
    //     int A_col_id=dptr_offset_A[block_row_start]+i;
    //     int A_col=dptr_colindex_A[A_col_id];
    //     //确定当前元素A的行
    //     for(int j=A_row;j<block_row_end;j++){
    //         if (A_col_id>=dptr_offset_A[A_row]&&A_col_id<dptr_offset_A[A_row+1]){
    //             break;
    //         }else{
    //             A_row++;
    //         }
    //     }
    //     //获取B的行偏移及列索引并完成计算
    //     int B_row_start=dptr_offset_B[A_col];
    //     int B_row_end=dptr_offset_B[A_col+1];
    //     for(int j=B_row_start;j<B_row_end;j++){
    //         int C_col=dptr_colindex_B[j];
    //         atomicAdd(tmp_dev_c+A_row*dev_B_col+C_col,alpha*(dptr_value_A[A_col_id])*(dptr_value_B[j]));
    //     }
    // }
}

