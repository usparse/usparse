#ifndef SPGEMM_DENSE_SPGEMM_DENSE_KERNEL_H
#define SPGEMM_DENSE_SPGEMM_DENSE_KERNEL_H
#include "types.hpp"

//每个THREAD计算A的一行
__global__ void division_by_row_thread_per_row(const csrValueType alpha,const size_t dev_A_row,const size_t dev_B_col,const size_t dev_A_col,const csrIndexType* dptr_offset_A,
const csrIndexType* dptr_colindex_A,const csrValueType* dptr_value_A,const csrIndexType* dptr_offset_B,
const csrIndexType* dptr_colindex_B,const csrValueType* dptr_value_B,csrValueType* tmp_dev_c);


//每个BLOCK计算A的若干行，每个thread共同处理一行
__global__ void division_by_row_thread_per_nnz(const xtype alpha,const size_t dev_A_row,const size_t dev_B_col,const size_t dev_A_col,const csrIndexType *dptr_offset_A,const csrIndexType *dptr_colindex_A,
const csrValueType *dptr_value_A,const csrIndexType *dptr_offset_B,const csrIndexType *dptr_colindex_B,
const csrValueType *dptr_value_B,xtype *tmp_dev_c);

//每个BLOCK计算A的若干行，每个wavefront处理一个nnzA
template<int WF_SIZE>
__global__ void division_by_row_wf_per_nnz(const xtype alpha,const size_t dev_A_row,const size_t dev_B_col,const size_t dev_A_col,const csrIndexType *dptr_offset_A,const csrIndexType *dptr_colindex_A,const csrValueType *dptr_value_A,const csrIndexType *dptr_offset_B,const csrIndexType *dptr_colindex_B,const csrValueType *dptr_value_B,
xtype *tmp_dev_c){

    const int wavefront_num = blockDim.x / WF_SIZE;
    const int wavefront_index = threadIdx.x / WF_SIZE;
    const int lane_index      = threadIdx.x % WF_SIZE;

    int rowA_per_Block=dev_A_row/gridDim.x;
    int reality_rowA_per_Block=rowA_per_Block;

    if(blockIdx.x<dev_A_row%gridDim.x){
        reality_rowA_per_Block++;
    }

    int block_row_start= blockIdx.x<dev_A_row%gridDim.x ? blockIdx.x*reality_rowA_per_Block : (dev_A_row%gridDim.x)*(reality_rowA_per_Block+1)+(blockIdx.x-dev_A_row%gridDim.x)*reality_rowA_per_Block;
    int block_row_end= blockIdx.x<dev_A_row%gridDim.x ? (blockIdx.x+1)*reality_rowA_per_Block : (dev_A_row%gridDim.x)*(reality_rowA_per_Block+1)+(blockIdx.x-dev_A_row%gridDim.x+1)*reality_rowA_per_Block;


    // 计算每个block需要计算的nnz数量
    int block_nnz=dptr_offset_A[block_row_end]-dptr_offset_A[block_row_start];
    
    extern __shared__ int s_offset_A[];
    
    for(int i=threadIdx.x;i<reality_rowA_per_Block+1;i=i+blockDim.x){
        s_offset_A[i]=dptr_offset_A[block_row_start+i];
    }

    __syncthreads();

    int s_A_row=0;//记录在LDS中的行偏移
    for(int i=wavefront_index;i<block_nnz;i=i+wavefront_num){
        int A_col_id=s_offset_A[0]+i;
        int A_col=dptr_colindex_A[A_col_id];
        //csrValueType A_val=dptr_value_A[A_col_id];
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
        for(int j=lane_index+B_row_start;j<B_row_end;j=j+WF_SIZE){
            int C_col=dptr_colindex_B[j];
            atomicAdd(tmp_dev_c+A_row*dev_B_col+C_col,alpha*(dptr_value_A[A_col_id])*(dptr_value_B[j]));
        }
    }
};

//每个BLOCK计算A的若干行，每个wavefront共同处理一个nnzA
template<int THREADS,int WF_SIZE,int SHARED_BUFF_NUM>
__global__ void division_by_row_wf_per_nnzV2(const xtype alpha,const csrIndexType dev_A_row,const csrIndexType dev_B_col,
const csrIndexType *dptr_offset_A,const csrIndexType *dptr_colindex_A,
const csrValueType *dptr_value_A,const csrIndexType *dptr_offset_B,const csrIndexType *dptr_colindex_B,
const csrValueType *dptr_value_B,xtype *tmp_dev_c){
    
    const int wavefront_num = blockDim.x / WF_SIZE;
    const int wavefront_index = threadIdx.x / WF_SIZE;
    const int lane_index      = threadIdx.x % WF_SIZE;

    int rowA_per_Block=dev_A_row/gridDim.x;
    int reality_rowA_per_Block=rowA_per_Block;

    if(blockIdx.x<dev_A_row%gridDim.x){
        reality_rowA_per_Block++;
    }

    int block_row_start= blockIdx.x<dev_A_row%gridDim.x ? blockIdx.x*reality_rowA_per_Block : (dev_A_row%gridDim.x)*(reality_rowA_per_Block+1)+(blockIdx.x-dev_A_row%gridDim.x)*reality_rowA_per_Block;
    int block_row_end= blockIdx.x<dev_A_row%gridDim.x ? (blockIdx.x+1)*reality_rowA_per_Block : (dev_A_row%gridDim.x)*(reality_rowA_per_Block+1)+(blockIdx.x-dev_A_row%gridDim.x+1)*reality_rowA_per_Block;
    
    extern __shared__ int s_offset_A[];
    for(int i=threadIdx.x;i<reality_rowA_per_Block+1;i=i+blockDim.x){
        s_offset_A[i]=dptr_offset_A[block_row_start+i];
    }
    __syncthreads();

    for(int i=0;i<reality_rowA_per_Block;i++){
        if(s_offset_A[i+1]-s_offset_A[i]>0){
            for(int j=s_offset_A[i]+wavefront_index;j<s_offset_A[i+1];j+=wavefront_num){
                int A_col=dptr_colindex_A[j];
                int B_row_start=dptr_offset_B[A_col];
                int B_row_end=dptr_offset_B[A_col+1];
                for(int k=lane_index+B_row_start;k<B_row_end;k=k+WF_SIZE){
                    int C_col=dptr_colindex_B[k];
                    atomicAdd(tmp_dev_c+(i+block_row_start)*dev_B_col+C_col,alpha*(dptr_value_A[j])*(dptr_value_B[k]));
                }
            }
        }else{
            continue;
        }
    }
};

#endif //SPGEMM_DENSE_SPGEMM_DENSE_KERNEL_H