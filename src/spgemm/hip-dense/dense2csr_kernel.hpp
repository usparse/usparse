#ifndef SPGEMM_DENSE2CSR_KERNEL_HPP
#define SPGEMM_DENSE2CSR_KERNEL_HPP

#include "building_config.h"

template <typename T, typename I,int NUMROWS_PER_BLOCK>
__global__  void dense2csr_kernel(const size_t m,
                          const size_t n,
                          const T* dense_val,
                          const size_t ld,
                          T* csr_val,
                          I* csr_row_ptr,
                          I* csr_col_ind)
{
    const int wavefront_index = hipThreadIdx_x / __WF_SIZE__;
    const int lane_index      = hipThreadIdx_x % __WF_SIZE__;
    const uint64_t  filter          = 0xffffffffffffffff >> (63 - lane_index);
    const I             row_index= NUMROWS_PER_BLOCK * hipBlockIdx_x + wavefront_index;

    if(row_index < m)
    {
        I shift = csr_row_ptr[row_index];
        //
        // The warp handles the entire row.
        //
        for(int column_index = lane_index; column_index < n; column_index += __WF_SIZE__)
        {
            //
            // Synchronize for cache considerations.
            //
            __syncthreads();

            //
            // Get value.
            //
            T value = static_cast<T>(0);

            value = dense_val[column_index + row_index * ld];

            //
            // Predicate.
            //
            const bool predicate = value != 0;

            //
            // Mask of the wavefront.
            //
            const uint64_t wavefront_mask = __ballot(predicate);

            //
            // Get the number of previous non-zero in the row.
            //
            const uint64_t count_previous_nnzs = __popcll(wavefront_mask & filter);

            if(predicate)
            {
                //
                // Calculate local index.
                //
                const uint64_t local_index_in_warp = count_previous_nnzs - 1;

                //
                // Populate the sparse matrix.
                //
                csr_val[shift + local_index_in_warp]     = value;
                csr_col_ind[shift + local_index_in_warp] = column_index;
            }

            //
            // Broadcast the update of the shift to all 64 threads for the next set of 64 columns.
            // Choose the last lane since that it contains the size of the sparse row (even if its predicate is false).
            //
            shift += __shfl(static_cast<int>(count_previous_nnzs), __WF_SIZE__ - 1);
        }
    }
}

template <typename T, typename I>
__global__ void dense2csr_kernel(const size_t m,
                          const size_t n,
                          const I* dense_col,
                          const T* dense_val,
                          T* csr_val,
                          I* csr_row_ptr,
                          I* csr_col_ind){
    
    const int row_a=blockIdx.x;
    const int row_start=csr_row_ptr[row_a];
    const int row_end=csr_row_ptr[row_a+1];
    for(int i=threadIdx.x;i<row_end-row_start;i+=blockDim.x){
        csr_col_ind[row_start+i]=dense_col[row_a*n+i];
        csr_val[row_start+i]=dense_val[row_a*n+i];
    }

    // const int row_a=blockIdx.x;
    // const int row_start=csr_row_ptr[row_a];
    // const int row_end=csr_row_ptr[row_a+1];
    // const int nnz=row_end-row_start;
    // const int loop_col=nnz/4*4;
    // const int loop_val=nnz/2*2;
    // for(int i=threadIdx.x*4;i<loop_col;i+=blockDim.x*4){
    //     csr_col_ind[row_start+i]=dense_col[row_a*n+i];
    //     csr_col_ind[row_start+i+1]=dense_col[row_a*n+i+1];
    //     csr_col_ind[row_start+i+2]=dense_col[row_a*n+i+2];
    //     csr_col_ind[row_start+i+3]=dense_col[row_a*n+i+3];
    //     // csr_val[row_start+i]=dense_val[row_a*n+i];
    // }
    // // __syncthreads();
    // if(nnz%4!=0){
    //     for(int i=threadIdx.x+loop_col;i<nnz;i+=blockDim.x){
    //         csr_col_ind[row_start+i]=dense_col[row_a*n+i];
    //     }
    // }
    // // __syncthreads();
    // for(int i=threadIdx.x*2;i<loop_val;i+=blockDim.x*2){
    //     csr_val[row_start+i]=dense_val[row_a*n+i];
    //     csr_val[row_start+i+1]=dense_val[row_a*n+i+1];
    // }
    // // __syncthreads();
    // if(nnz%2!=0){
    //     for(int i=threadIdx.x+loop_val;i<nnz;i+=blockDim.x){
    //         csr_col_ind[row_start+i]=dense_col[row_a*n+i];
    //     }
    // }
    // __syncthreads();
}

#endif // SPGEMM_DENSE2CSR_KERNEL_HPP
