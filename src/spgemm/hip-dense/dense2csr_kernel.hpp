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

#endif // SPGEMM_DENSE2CSR_KERNEL_HPP
