#ifndef SPGEMM_DENSE_SPGEMM_DENSE_KERNEL_H
#define SPGEMM_DENSE_SPGEMM_DENSE_KERNEL_H

#include "building_config.h"
#include "stdio.h"


//每个block处理A的若干行，每次处理一行，每个wavefront计算一个nnzA
//使用LDS缓存该行计算结果（结果分多次计算得到，每次计算CHUNK_SIZE个结果）
//每行最后一次分配nnzA时无法保证每个wavefront分配到一个nnzA
template<typename T, typename I,size_t CHUNK_SIZE,size_t THREADS>
__global__ void wf_per_nnzA_kernel(const T alpha,const size_t dev_A_row,const size_t dev_B_col,const size_t dev_A_col,
const I *dptr_offset_A,const I *dptr_colindex_A,const T *dptr_value_A,const I *dptr_offset_B,const I *dptr_colindex_B,
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
    
    __shared__ T s_tmp_dev_c[CHUNK_SIZE];
    for(int i=threadIdx.x;i<CHUNK_SIZE;i+=blockDim.x){
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
                int B_loop_counter=dev_B_col%(CHUNK_SIZE)==0?dev_B_col/(CHUNK_SIZE):dev_B_col/(CHUNK_SIZE)+1;
                int k=lane_index+B_row_start;
                for(int B_loop=0;B_loop<B_loop_counter;B_loop++){
                    for(;k<B_row_end;k+=__WF_SIZE__){
                        int C_col=dptr_colindex_B[k];
                        if(C_col<(B_loop+1)*(CHUNK_SIZE)&&C_col<dev_B_col){
                            atomicAdd(s_tmp_dev_c+C_col%CHUNK_SIZE,alpha*(dptr_value_A[j])*(dptr_value_B[k]));
                        }else{
                            break;
                        }
                    }
                    __syncthreads();
                    int transfer_up_bound=B_loop==B_loop_counter-1?dev_B_col-B_loop*CHUNK_SIZE:CHUNK_SIZE;
                    if(transfer_up_bound%2==0){
                        for(int transfer=threadIdx.x*2;transfer<transfer_up_bound-1;transfer+=blockDim.x*2){
                            tmp_dev_c[(i+block_row_start)*dev_B_col+B_loop*CHUNK_SIZE+transfer]+=s_tmp_dev_c[transfer];
                            tmp_dev_c[(i+block_row_start)*dev_B_col+B_loop*CHUNK_SIZE+transfer+1]+=s_tmp_dev_c[transfer+1];
                        }
                    }else{
                        for(int transfer=threadIdx.x;transfer<transfer_up_bound;transfer+=blockDim.x){
                            tmp_dev_c[(i+block_row_start)*dev_B_col+B_loop*CHUNK_SIZE+transfer]+=s_tmp_dev_c[transfer];
                        }
                    }
                    __syncthreads();
                    for(int transfer=threadIdx.x;transfer<CHUNK_SIZE;transfer+=blockDim.x){
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
                int B_loop_counter=dev_B_col%CHUNK_SIZE==0?dev_B_col/CHUNK_SIZE:dev_B_col/CHUNK_SIZE+1;
                for(int B_loop=0;B_loop<B_loop_counter;B_loop++){
                    if(j<A_row_end){
                        int B_row_end=dptr_offset_B[dptr_colindex_A[j]+1];
                        for(;k<B_row_end;k+=__WF_SIZE__){
                            int C_col=dptr_colindex_B[k];
                            if(C_col<(B_loop+1)*CHUNK_SIZE&&C_col<dev_B_col){
                                atomicAdd(s_tmp_dev_c+C_col%CHUNK_SIZE,alpha*(dptr_value_A[j])*(dptr_value_B[k]));
                            }else{
                                break;
                            }
                        }
                    }
                    __syncthreads();
                    int transfer_up_bound=B_loop==B_loop_counter-1?dev_B_col-B_loop*CHUNK_SIZE:CHUNK_SIZE;
                    if(transfer_up_bound%2==0){
                        for(int transfer=threadIdx.x*2;transfer<transfer_up_bound-1;transfer+=blockDim.x*2){
                            tmp_dev_c[(i+block_row_start)*dev_B_col+B_loop*CHUNK_SIZE+transfer]+=s_tmp_dev_c[transfer];
                            tmp_dev_c[(i+block_row_start)*dev_B_col+B_loop*CHUNK_SIZE+transfer+1]+=s_tmp_dev_c[transfer+1];
                        }
                    }else{
                        for(int transfer=threadIdx.x;transfer<transfer_up_bound;transfer+=blockDim.x){
                            tmp_dev_c[(i+block_row_start)*dev_B_col+B_loop*CHUNK_SIZE+transfer]+=s_tmp_dev_c[transfer];
                        }
                    }
                    __syncthreads();
                    for(int transfer=threadIdx.x;transfer<CHUNK_SIZE;transfer+=blockDim.x){
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

//借鉴rocsparse的multipass策略中前半部分计算稠密结果的策略(wf_per_nnzA_kernel_multipass)
template<typename T, typename I,size_t CHUNK_SIZE,size_t THREADS, size_t VEC_SIZE>
__global__ void wf_per_nnzA_kernel(const T alpha,const size_t dev_A_row,const size_t dev_B_col,const size_t dev_A_col,
const I *dptr_offset_A,const I *dptr_colindex_A,const T *dptr_value_A,const I *dptr_offset_B,const I *dptr_colindex_B,
const T *dptr_value_B,T *tmp_dev_c,I *workspace_b){

    const int block_vec_id = threadIdx.x / VEC_SIZE;
    constexpr int block_vec_num = THREADS / VEC_SIZE;
    const int vec_thread_id = threadIdx.x % VEC_SIZE;

    __shared__ T s_tmp_dev_c[CHUNK_SIZE];
    __shared__ I next_chunk;

    const int row_a = blockIdx.x;

    if(row_a>=dev_A_row){
        return;
    }

    const I row_idx_begin = dptr_offset_A[row_a];
    const I row_idx_end = dptr_offset_A[row_a + 1];
    int chunk_begin = 0;
    int chunk_end = CHUNK_SIZE;
    while (chunk_begin<dev_B_col){
        // init
        for (int i = threadIdx.x; i < CHUNK_SIZE; i += blockDim.x) {
            s_tmp_dev_c[i] = static_cast<T>(0);
        }
        if(threadIdx.x == 0){
            next_chunk = dev_B_col;
        }
        __syncthreads();

        I min_col = dev_B_col;

        for (I idx_a = row_idx_begin + block_vec_id; idx_a < row_idx_end; idx_a += block_vec_num) {
            const I col_a = dptr_colindex_A[idx_a];
            const I row_b = col_a;
            const T val_A = alpha * dptr_value_A[idx_a];

            I idx_begin_b = (chunk_begin == 0) ? dptr_offset_B[row_b] : workspace_b[idx_a];
            I idx_end_b = dptr_offset_B[row_b + 1];

            I next_k = idx_begin_b + vec_thread_id;
            for (I idx_b = next_k; idx_b < idx_end_b; idx_b += VEC_SIZE) {
                const I col_b = dptr_colindex_B[idx_b];
                if (col_b >= chunk_begin && col_b < chunk_end) {
                    atomicAdd(s_tmp_dev_c + col_b - chunk_begin, val_A * dptr_value_B[idx_b]);
                } else if (col_b >= chunk_end) {
                    next_k = idx_b;
                    min_col = min(min_col, col_b);
                    break;
                }
            }
            for (int i = 1; i < VEC_SIZE; i <<= 1) {
                next_k = min(next_k, __shfl_up(next_k, i, VEC_SIZE));
            }

            if (vec_thread_id == VEC_SIZE - 1) {
                workspace_b[idx_a] = next_k;
            }
        }
        
        for (int i = 1; i < VEC_SIZE; i <<= 1) {
            min_col = min(min_col, __shfl_up(min_col, i, VEC_SIZE));
        }

        if (vec_thread_id == VEC_SIZE - 1) {
            atomicMin(&next_chunk, min_col);
        }
        __syncthreads();
        int transfer_up_bound=(dev_B_col-chunk_begin)<=CHUNK_SIZE?dev_B_col-chunk_begin:CHUNK_SIZE;
        if(transfer_up_bound%2==0){
            for(int transfer=threadIdx.x*2;transfer<transfer_up_bound-1;transfer+=blockDim.x*2){
                tmp_dev_c[blockIdx.x*dev_B_col+chunk_begin+transfer]=s_tmp_dev_c[transfer];
                tmp_dev_c[blockIdx.x*dev_B_col+chunk_begin+transfer+1]=s_tmp_dev_c[transfer+1];
            }
        }else{
            for(int transfer=threadIdx.x;transfer<transfer_up_bound;transfer+=blockDim.x){
                tmp_dev_c[blockIdx.x*dev_B_col+chunk_begin+transfer]=s_tmp_dev_c[transfer];
            }
        }
        __syncthreads();
        chunk_begin = next_chunk;
        chunk_end = chunk_begin + CHUNK_SIZE;
        __syncthreads();
    }
};

//通过rocsparse的multipass修改得到的dense2csr策略(dense_multipass)
template <typename T, typename I, size_t CHUNK_SIZE, size_t THREADS, size_t VEC_SIZE>
__global__ void __launch_bounds__(THREADS)
    wf_per_nnzA_kernel(const T alpha,const size_t dev_A_row,const size_t dev_B_col,const size_t dev_A_col,
const I *dptr_offset_A,const I *dptr_colindex_A,const T *dptr_value_A,const I *dptr_offset_B,const I *dptr_colindex_B,
const T *dptr_value_B,I* dptr_offset_c,I* dense_col,T *dense_val,I *workspace_b) {
    
  // vec id in block
  const int block_vec_id = threadIdx.x / VEC_SIZE;
  // vec num in block
  constexpr int block_vec_num = THREADS / VEC_SIZE;
  // thread id in vec
  const int vec_thread_id = threadIdx.x % VEC_SIZE;

  // chunks
  __shared__ bool table[CHUNK_SIZE];
  __shared__ T data[CHUNK_SIZE];

  __shared__ I next_chunk;
  // block processes a row
  const int row_a = blockIdx.x;
  __shared__ int row_a_nnzc;
  row_a_nnzc=0;
    // __shared__ int row_a_nnz[THREADS];


  if (row_a >= dev_A_row) {
    return;
  }

  const I row_idx_begin = dptr_offset_A[row_a];
  const I row_idx_end = dptr_offset_A[row_a + 1];
  I row_begin_c = blockIdx.x*dev_B_col;

  int chunk_begin = 0;
  int chunk_end = CHUNK_SIZE;

  while (chunk_begin < dev_B_col) {
    // init
    for (int i = threadIdx.x; i < CHUNK_SIZE; i += THREADS) {
      table[i] = 0;
      data[i] = static_cast<T>(0);
    }
    if (threadIdx.x == 0) {
      next_chunk = dev_B_col;
    }
    __syncthreads();
    I min_col = dev_B_col;

    for (I idx_a = row_idx_begin + block_vec_id; idx_a < row_idx_end; idx_a += block_vec_num) {
      const I col_a = dptr_colindex_A[idx_a];
      const I row_b = col_a;
      const T val_A = alpha * dptr_value_A[idx_a];

      I idx_begin_b = (chunk_begin == 0) ? dptr_offset_B[row_b] : workspace_b[idx_a];
      I idx_end_b = dptr_offset_B[row_b + 1];

      I next_k = idx_begin_b + vec_thread_id;
      for (I idx_b = next_k; idx_b < idx_end_b; idx_b += VEC_SIZE) {
        const I col_b = dptr_colindex_B[idx_b];
        if (col_b >= chunk_begin && col_b < chunk_end) {
          table[col_b - chunk_begin] = 1;
          atomicAdd(data + col_b - chunk_begin, val_A * dptr_value_B[idx_b]);
        } else if (col_b >= chunk_end) {
          next_k = idx_b;
          min_col = min(min_col, col_b);
          break;
        }
      }
      // rocsparse_wfreduce_min<VEC_SIZE>(&next_k);
      for (int i = 1; i < VEC_SIZE; i <<= 1) {
        next_k = min(next_k, __shfl_up(next_k, i, VEC_SIZE));
      }

      if (vec_thread_id == VEC_SIZE - 1) {
        workspace_b[idx_a] = next_k;
      }
    }
    // rocsparse_wfreduce_min<VEC_SIZE>(&min_col);
    for (int i = 1; i < VEC_SIZE; i <<= 1) {
      min_col = min(min_col, __shfl_up(min_col, i, VEC_SIZE));
    }

    if (vec_thread_id == VEC_SIZE - 1) {
      atomicMin(&next_chunk, min_col);
    }
    __syncthreads();
    // re-use shared memory
    int *scan_offsets = reinterpret_cast<int *>(data);
    // compress
    for (int i = threadIdx.x; i < CHUNK_SIZE; i += THREADS) {
      bool has_nnz = table[i];
      T value = data[i];

      unsigned long long mask = __ballot(has_nnz);
      int nnz = __popcll(mask);

      unsigned long long lanemask_le = UINT64_MAX >> (sizeof(unsigned long long) * CHAR_BIT - (__lane_id() + 1));

      int offset = __popcll(lanemask_le & mask);
      __syncthreads();

      scan_offsets[threadIdx.x / __WF_SIZE__] = nnz;
      __syncthreads();

      for (unsigned int j = 1; j < THREADS / __WF_SIZE__; ++j) {
        if (threadIdx.x >= j * __WF_SIZE__) {
          offset += scan_offsets[j - 1];
        }
      }
      I idx = row_begin_c + offset - 1;

      if (has_nnz) {
        atomicAdd(&row_a_nnzc, 1);
        // nnzc_row++;
        dense_col[idx] = chunk_begin + i;
        dense_val[idx] = value;
      }

      if (threadIdx.x == THREADS - 1) {
        scan_offsets[THREADS / __WF_SIZE__ - 1] = offset;
      }
      __syncthreads();
      row_begin_c += scan_offsets[THREADS / __WF_SIZE__ - 1];
    }
    chunk_begin = next_chunk;
    chunk_end = chunk_begin + CHUNK_SIZE;
    __syncthreads();
  }
  if(threadIdx.x==0){
    dptr_offset_c[row_a+1]=row_a_nnzc;
  }
};

#endif //SPGEMM_DENSE_SPGEMM_DENSE_KERNEL_H