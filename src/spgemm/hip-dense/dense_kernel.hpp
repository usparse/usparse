#ifndef SPGEMM_DENSE_SPGEMM_DENSE_KERNEL_H
#define SPGEMM_DENSE_SPGEMM_DENSE_KERNEL_H

#include "building_config.h"

//每个block处理A的若干行，每次处理一行，每个wavefront计算一个nnzA
//当每个wavefront均能分配到nnzA时使用LDS缓存该行计算结果（结果分多次计算得到，每次计算THREADS*SHARED_BUFF_NUM个结果）
template<typename T, typename I,int THREADS,int SHARED_BUFF_NUM>
__global__ void wf_per_nnzA_kernel1(const T alpha,const size_t dev_A_row,const size_t dev_B_col,const size_t dev_A_col,
const I *dptr_offset_A,const I *dptr_colindex_A,
const T *dptr_value_A,const I *dptr_offset_B,const I *dptr_colindex_B,
const T *dptr_value_B,T *tmp_dev_c){
    
    const int wavefront_num = blockDim.x / __WF_SIZE__;
    const int wavefront_index = threadIdx.x / __WF_SIZE__;
    const int lane_index      = threadIdx.x % __WF_SIZE__;

    int rowA_per_Block=dev_A_row/gridDim.x;
    int reality_rowA_per_Block=rowA_per_Block;

    if(blockIdx.x<dev_A_row%gridDim.x){
        reality_rowA_per_Block++;
    }

    int block_row_start= blockIdx.x<dev_A_row%gridDim.x ? blockIdx.x*reality_rowA_per_Block : (dev_A_row%gridDim.x)*(reality_rowA_per_Block+1)+(blockIdx.x-dev_A_row%gridDim.x)*reality_rowA_per_Block;
    int block_row_end= blockIdx.x<dev_A_row%gridDim.x ? (blockIdx.x+1)*reality_rowA_per_Block : (dev_A_row%gridDim.x)*(reality_rowA_per_Block+1)+(blockIdx.x-dev_A_row%gridDim.x+1)*reality_rowA_per_Block;
    

    __shared__ T s_tmp_dev_c[THREADS*SHARED_BUFF_NUM];
    for(int i=threadIdx.x;i<THREADS*SHARED_BUFF_NUM;i+=blockDim.x){
        s_tmp_dev_c[i]=0;
    }
    __syncthreads();

    for(int i=0;i<reality_rowA_per_Block;i++){
        int A_row_start=dptr_offset_A[i+block_row_start];
        int A_row_end=dptr_offset_A[i+block_row_start+1];
        int A_row_nnz=A_row_end-A_row_start;
        if(A_row_nnz>0){
            int A_loop_counter=A_row_nnz/wavefront_num;
            //首先进行A_loop_counter次能填满每个wavefront的计算
            for(int A_loop=0;A_loop<A_loop_counter;A_loop++){
                int j=wavefront_index+A_row_start+A_loop*wavefront_num;
                int A_col=dptr_colindex_A[j];
                int B_row_start=dptr_offset_B[A_col];
                int B_row_end=dptr_offset_B[A_col+1];
                int B_loop_counter=dev_B_col%(THREADS*SHARED_BUFF_NUM)==0?dev_B_col/(THREADS*SHARED_BUFF_NUM):dev_B_col/(THREADS*SHARED_BUFF_NUM)+1;
                int k=lane_index+B_row_start;
                for(int B_loop=0;B_loop<B_loop_counter;B_loop++){
                    for(;k<B_row_end;k+=__WF_SIZE__){
                        int C_col=dptr_colindex_B[k];
                        if(C_col<(B_loop+1)*(THREADS*SHARED_BUFF_NUM)&&C_col<dev_B_col){
                            atomicAdd(s_tmp_dev_c+C_col%(THREADS*SHARED_BUFF_NUM),alpha*(dptr_value_A[j])*(dptr_value_B[k]));
                        }else{
                            break;
                        }
                    }
                    __syncthreads();
                    int transfer_up_bound=B_loop==B_loop_counter-1?dev_B_col-B_loop*(THREADS*SHARED_BUFF_NUM):THREADS*SHARED_BUFF_NUM;
                    if(SHARED_BUFF_NUM%2==0){
                        for(int transfer=threadIdx.x*2;transfer<transfer_up_bound-1;transfer+=blockDim.x*2){
                            tmp_dev_c[(i+block_row_start)*dev_B_col+B_loop*(THREADS*SHARED_BUFF_NUM)+transfer]+=s_tmp_dev_c[transfer];
                            tmp_dev_c[(i+block_row_start)*dev_B_col+B_loop*(THREADS*SHARED_BUFF_NUM)+transfer+1]+=s_tmp_dev_c[transfer+1];
                        }
                    }else{
                        for(int transfer=threadIdx.x;transfer<transfer_up_bound;transfer+=blockDim.x){
                            tmp_dev_c[(i+block_row_start)*dev_B_col+B_loop*(THREADS*SHARED_BUFF_NUM)+transfer]+=s_tmp_dev_c[transfer];
                        }
                    }
                    __syncthreads();
                    for(int transfer=threadIdx.x;transfer<THREADS*SHARED_BUFF_NUM;transfer+=blockDim.x){
                        s_tmp_dev_c[transfer]=0;
                    }
                    __syncthreads();
                }
            }
            //处理此行最终不足wavefront_num个非零元素
            if(A_row_nnz%wavefront_num!=0){
                for(int j=wavefront_index/2+A_row_start+A_loop_counter*wavefront_num;j<A_row_end;j+=wavefront_num/2){
                    int A_col=dptr_colindex_A[j];
                    int B_row_start=dptr_offset_B[A_col];
                    int B_row_end=dptr_offset_B[A_col+1];
                    for(int k=lane_index+B_row_start+(wavefront_index%2)*__WF_SIZE__;k<B_row_end;k=k+__WF_SIZE__*2){
                        int C_col=dptr_colindex_B[k];
                        atomicAdd(tmp_dev_c+(i+block_row_start)*dev_B_col+C_col,alpha*(dptr_value_A[j])*(dptr_value_B[k]));
                    }
                }
            }
            __syncthreads();
        }else{
            continue;
        }
    }
};


//每个block处理A的若干行，每次处理一行，每个wavefront计算一个nnzA
//使用LDS缓存该行计算结果（结果分多次计算得到，每次计算THREADS*SHARED_BUFF_NUM个结果）
//每行最后一次分配nnzA时无法保证每个wavefront分配到一个nnzA
template<typename T, typename I,int THREADS,int SHARED_BUFF_NUM>
__global__ void wf_per_nnzA_kernel2(const T alpha,const size_t dev_A_row,const size_t dev_B_col,const size_t dev_A_col,
const I *dptr_offset_A,const I *dptr_colindex_A,
const T *dptr_value_A,const I *dptr_offset_B,const I *dptr_colindex_B,
const T *dptr_value_B,T *tmp_dev_c){
    
    const int wavefront_num = blockDim.x / __WF_SIZE__;
    const int wavefront_index = threadIdx.x / __WF_SIZE__;
    const int lane_index      = threadIdx.x % __WF_SIZE__;

    int rowA_per_Block=dev_A_row/gridDim.x;
    int reality_rowA_per_Block=rowA_per_Block;

    if(blockIdx.x<dev_A_row%gridDim.x){
        reality_rowA_per_Block++;
    }

    int block_row_start= blockIdx.x<dev_A_row%gridDim.x ? blockIdx.x*reality_rowA_per_Block : (dev_A_row%gridDim.x)*(reality_rowA_per_Block+1)+(blockIdx.x-dev_A_row%gridDim.x)*reality_rowA_per_Block;
    int block_row_end= blockIdx.x<dev_A_row%gridDim.x ? (blockIdx.x+1)*reality_rowA_per_Block : (dev_A_row%gridDim.x)*(reality_rowA_per_Block+1)+(blockIdx.x-dev_A_row%gridDim.x+1)*reality_rowA_per_Block;
    
    __shared__ T s_tmp_dev_c[THREADS*SHARED_BUFF_NUM];
    for(int i=threadIdx.x;i<THREADS*SHARED_BUFF_NUM;i+=blockDim.x){
        s_tmp_dev_c[i]=0;
    }
    __syncthreads();

    for(int i=0;i<reality_rowA_per_Block;i++){
        int A_row_start=dptr_offset_A[i+block_row_start];
        int A_row_end=dptr_offset_A[i+block_row_start+1];
        int A_row_nnz=A_row_end-A_row_start;
        if(A_row_nnz>0){
            int A_loop_counter=A_row_nnz/wavefront_num;
            //首先进行A_loop_counter次能填满每个wavefront的计算
            for(int A_loop=0;A_loop<A_loop_counter;A_loop++){
                int j=wavefront_index+A_row_start+A_loop*wavefront_num;
                int A_col=dptr_colindex_A[j];
                int B_row_start=dptr_offset_B[A_col];
                int B_row_end=dptr_offset_B[A_col+1];
                int B_loop_counter=dev_B_col%(THREADS*SHARED_BUFF_NUM)==0?dev_B_col/(THREADS*SHARED_BUFF_NUM):dev_B_col/(THREADS*SHARED_BUFF_NUM)+1;
                int k=lane_index+B_row_start;
                for(int B_loop=0;B_loop<B_loop_counter;B_loop++){
                    for(;k<B_row_end;k+=__WF_SIZE__){
                        int C_col=dptr_colindex_B[k];
                        if(C_col<(B_loop+1)*(THREADS*SHARED_BUFF_NUM)&&C_col<dev_B_col){
                            atomicAdd(s_tmp_dev_c+C_col%(THREADS*SHARED_BUFF_NUM),alpha*(dptr_value_A[j])*(dptr_value_B[k]));
                        }else{
                            break;
                        }
                    }
                    __syncthreads();
                    int transfer_up_bound=B_loop==B_loop_counter-1?dev_B_col-B_loop*(THREADS*SHARED_BUFF_NUM):THREADS*SHARED_BUFF_NUM;
                    if(SHARED_BUFF_NUM%2==0){
                        for(int transfer=threadIdx.x*2;transfer<transfer_up_bound-1;transfer+=blockDim.x*2){
                            tmp_dev_c[(i+block_row_start)*dev_B_col+B_loop*(THREADS*SHARED_BUFF_NUM)+transfer]+=s_tmp_dev_c[transfer];
                            tmp_dev_c[(i+block_row_start)*dev_B_col+B_loop*(THREADS*SHARED_BUFF_NUM)+transfer+1]+=s_tmp_dev_c[transfer+1];
                        }
                    }else{
                        for(int transfer=threadIdx.x;transfer<transfer_up_bound;transfer+=blockDim.x){
                            tmp_dev_c[(i+block_row_start)*dev_B_col+B_loop*(THREADS*SHARED_BUFF_NUM)+transfer]+=s_tmp_dev_c[transfer];
                        }
                    }
                    __syncthreads();
                    for(int transfer=threadIdx.x;transfer<THREADS*SHARED_BUFF_NUM;transfer+=blockDim.x){
                        s_tmp_dev_c[transfer]=0;
                    }
                    __syncthreads();
                }
            }
            //处理此行最终不足wavefront_num个非零元素
            if(A_row_nnz%wavefront_num!=0){
                int j=wavefront_index+A_row_start+A_loop_counter*wavefront_num;
                int k=0;
                if(j<A_row_end){
                    k=lane_index+dptr_offset_B[dptr_colindex_A[j]];
                }
                int B_loop_counter=dev_B_col%(THREADS*SHARED_BUFF_NUM)==0?dev_B_col/(THREADS*SHARED_BUFF_NUM):dev_B_col/(THREADS*SHARED_BUFF_NUM)+1;
                for(int B_loop=0;B_loop<B_loop_counter;B_loop++){
                    if(j<A_row_end){
                        int B_row_end=dptr_offset_B[dptr_colindex_A[j]+1];
                        for(;k<B_row_end;k+=__WF_SIZE__){
                            int C_col=dptr_colindex_B[k];
                            if(C_col<(B_loop+1)*(THREADS*SHARED_BUFF_NUM)&&C_col<dev_B_col){
                                atomicAdd(s_tmp_dev_c+C_col%(THREADS*SHARED_BUFF_NUM),alpha*(dptr_value_A[j])*(dptr_value_B[k]));
                            }else{
                                break;
                            }
                        }
                    }
                    __syncthreads();
                    int transfer_up_bound=B_loop==B_loop_counter-1?dev_B_col-B_loop*(THREADS*SHARED_BUFF_NUM):THREADS*SHARED_BUFF_NUM;
                    if(SHARED_BUFF_NUM%2==0){
                        for(int transfer=threadIdx.x*2;transfer<transfer_up_bound-1;transfer+=blockDim.x*2){
                            tmp_dev_c[(i+block_row_start)*dev_B_col+B_loop*(THREADS*SHARED_BUFF_NUM)+transfer]+=s_tmp_dev_c[transfer];
                            tmp_dev_c[(i+block_row_start)*dev_B_col+B_loop*(THREADS*SHARED_BUFF_NUM)+transfer+1]+=s_tmp_dev_c[transfer+1];
                        }
                    }else{
                        for(int transfer=threadIdx.x;transfer<transfer_up_bound;transfer+=blockDim.x){
                            tmp_dev_c[(i+block_row_start)*dev_B_col+B_loop*(THREADS*SHARED_BUFF_NUM)+transfer]+=s_tmp_dev_c[transfer];
                        }
                    }
                    __syncthreads();
                    for(int transfer=threadIdx.x;transfer<THREADS*SHARED_BUFF_NUM;transfer+=blockDim.x){
                        s_tmp_dev_c[transfer]=0;
                    }
                    __syncthreads();
                }
            }
            __syncthreads();
        }else{
            continue;
        }
    }
};

#endif //SPGEMM_DENSE_SPGEMM_DENSE_KERNEL_H