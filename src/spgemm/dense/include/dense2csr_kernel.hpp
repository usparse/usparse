#pragma once
#include <stdio.h>

// template <int NUMROWS_PER_BLOCK,int WF_SIZE,typename I,typename J,typename T>
// __global__  void dense2csr_kernel(J m,
//                           J n,
//                           const T* dense_val,
//                           I ld,
//                           T* csr_val,
//                           I* csr_row_ptr,
//                           J* csr_col_ind)
// {
//     if(threadIdx.x==0&&blockIdx.x==0){
//         printf("22222222\n");
//     }
//     const int wavefront_index = hipThreadIdx_x / WF_SIZE;
//     const J             lane_index      = hipThreadIdx_x % WF_SIZE;
//     const uint64_t      filter          = 0xffffffffffffffff >> (63 - lane_index);
//     const J             row_index       = NUMROWS_PER_BLOCK * hipBlockIdx_x + wavefront_index;

//     if(row_index < m)
//     {
//         I shift = csr_row_ptr[row_index];
//         //
//         // The warp handles the entire row.
//         //
//         for(J column_index = lane_index; column_index < n; column_index += WF_SIZE)
//         {
//             //
//             // Synchronize for cache considerations.
//             //
//             __syncthreads();

//             //
//             // Get value.
//             //
//             T value = static_cast<T>(0);

//             value = dense_val[column_index + row_index * ld];

//             //
//             // Predicate.
//             //
//             const bool predicate = value != 0;

//             //
//             // Mask of the wavefront.
//             //
//             const uint64_t wavefront_mask = __ballot(predicate);

//             //
//             // Get the number of previous non-zero in the row.
//             //
//             const uint64_t count_previous_nnzs = __popcll(wavefront_mask & filter);

//             if(predicate)
//             {
//                 //
//                 // Calculate local index.
//                 //
//                 const uint64_t local_index_in_warp = count_previous_nnzs - 1;

//                 //
//                 // Populate the sparse matrix.
//                 //
//                 csr_val[shift + local_index_in_warp]     = value;
//                 csr_col_ind[shift + local_index_in_warp] = column_index;
//             }

//             //
//             // Broadcast the update of the shift to all 64 threads for the next set of 64 columns.
//             // Choose the last lane since that it contains the size of the sparse row (even if its predicate is false).
//             //
//             shift += __shfl(static_cast<int>(count_previous_nnzs), WF_SIZE - 1);
//         }
//     }
// }

// template <typename I, typename J, typename T>
// void dense2csx_template(J                         m,
//                         J                         n,
//                         const T*                  A,
//                         I                         ld,
//                         T*                        csx_val,
//                         I*                        csx_row_ptr,
//                         J*                        csx_col_ind)
// {
//     static constexpr int data_ratio = sizeof(T) / sizeof(float);
//     static constexpr int WF_SIZE         = 64;
//     static constexpr int NROWS_PER_BLOCK = 16 / (data_ratio > 0 ? data_ratio : 1);
//     int                  blocks          = (m - 1) / NROWS_PER_BLOCK + 1;
//     dim3                 k_blocks(blocks), k_threads(WF_SIZE * NROWS_PER_BLOCK);
//     std::cout<<"point111"<<std::endl;
//     dense2csr_kernel<NROWS_PER_BLOCK, WF_SIZE><<<k_blocks,k_threads>>>(m,n,A,ld,csx_val,csx_row_ptr,csx_col_ind);
// }

template <int NUMROWS_PER_BLOCK,int WF_SIZE>
__global__  void dense2csr_kernel(int m,
                          int n,
                          const dtype* dense_val,
                          int ld,
                          csrValueType* csr_val,
                          csrIndexType* csr_row_ptr,
                          csrIndexType* csr_col_ind)
{
    const int wavefront_index = hipThreadIdx_x / WF_SIZE;
    const int lane_index      = hipThreadIdx_x % WF_SIZE;
    const uint64_t  filter          = 0xffffffffffffffff >> (63 - lane_index);
    const csrIndexType             row_index= NUMROWS_PER_BLOCK * hipBlockIdx_x + wavefront_index;

    if(row_index < m)
    {
        csrIndexType shift = csr_row_ptr[row_index];
        //
        // The warp handles the entire row.
        //
        for(int column_index = lane_index; column_index < n; column_index += WF_SIZE)
        {
            //
            // Synchronize for cache considerations.
            //
            __syncthreads();

            //
            // Get value.
            //
            csrValueType value = static_cast<dtype>(0);

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
            shift += __shfl(static_cast<int>(count_previous_nnzs), WF_SIZE - 1);
        }
    }
}

void dense2csx_template(int                         m,
                        int                         n,
                        const dtype*                A,
                        int                         ld,
                        dtype*                      csx_val,
                        csrIdxType*                 csx_row_ptr,
                        csrIdxType*                 csx_col_ind)
{
    static constexpr int data_ratio = sizeof(dtype) / sizeof(float);
    static constexpr int WF_SIZE         = 64;
    static constexpr int NROWS_PER_BLOCK = 16 / (data_ratio > 0 ? data_ratio : 1);
    int                  blocks          = (m - 1) / NROWS_PER_BLOCK + 1;
    dim3                 k_blocks(blocks), k_threads(WF_SIZE * NROWS_PER_BLOCK);
    dense2csr_kernel<NROWS_PER_BLOCK, WF_SIZE><<<k_blocks,k_threads>>>(m,n,A,ld,csx_val,csx_row_ptr,csx_col_ind);
}