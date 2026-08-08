//
// Created by genshen to 2026-08-08.
//

#ifndef USPARSE_KERNELS__USPARSE_REDUCE_BY_KEY_HPP
#define USPARSE_KERNELS__USPARSE_REDUCE_BY_KEY_HPP

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

namespace usparse {
  namespace primitives {
    namespace reduce_by_key {
      template <typename INDEX_TYPE, typename Y_TYPE> struct KV {
        INDEX_TYPE key;
        Y_TYPE value;
      };

      namespace block {
        // segmented block reduction kernel, rows and vals are shared memory array, its length is BLOCK_SIZE.
        template <typename I, typename T, int BLOCK_SIZE>
        __device__ __forceinline__ void segmented_blockreduce(const I *rows, T *vals) {
          const int tid = hipThreadIdx_x;

#pragma unroll
          for (int j = 1; j < BLOCK_SIZE; j <<= 1) { // j is the step length
            T val = static_cast<T>(0);
            if (tid >= j) {
              if (rows[tid] == rows[tid - j]) { // with the same key.
                val = vals[tid - j];
              }
            }
            __syncthreads();

            vals[tid] = vals[tid] + val;
            __syncthreads();
          }
          return;
        }
      } // namespace block

      namespace device {
        // do the final block reduction of the block reduction buffers back into global memory
        template <uint32_t BLOCKSIZE, typename I, typename Y, typename T>
        __global__ __launch_bounds__(512) void final_segmented_loops_reduce_device(I nblocks,
                                                                                   const KV<I, T> *__restrict__ key_val,
                                                                                   Y *__restrict__ y) {
          const int tid = hipThreadIdx_x;

          // shared memory to hold row indices and values for segmented reduction
          __shared__ I shared_row[BLOCKSIZE];
          __shared__ T shared_val[BLOCKSIZE];

          // loop over blocks that are subject for segmented reduction
          for (I i = 0; i < nblocks; i += BLOCKSIZE) {
            // copy data to reduction buffers
            shared_row[tid] = (tid + i < nblocks) ? key_val[tid + i].key : -1;
            shared_val[tid] = (tid + i < nblocks) ? key_val[tid + i].value : static_cast<T>(0);

            __syncthreads();

            // do segmented block reduction
            usparse::primitives::reduce_by_key::block::segmented_blockreduce<I, T, BLOCKSIZE>(shared_row, shared_val);

            // add reduced sum to y if valid
            const I row = shared_row[tid];
            const I rowp1 = (tid < BLOCKSIZE - 1) ? shared_row[tid + 1] : -1;

            if (row != rowp1 && row >= 0) {
              y[row] = y[row] + shared_val[tid];
            }

            __syncthreads();
          }
          return;
        }

        template <uint32_t BLOCKSIZE, typename I, typename Y, typename T>
        __global__
        __launch_bounds__(512) void segmented_loops_reduce_device(I nblocks, const KV<I, T> *__restrict__ key_val,
                                                                  Y *__restrict__ y, KV<I, T> *__restrict__ temp_data) {
          const int tid = blockIdx.x * blockDim.x + threadIdx.x;
          const int tid_in_block = threadIdx.x;

          // shared memory to hold row indices and values for segmented reduction
          __shared__ I shared_row[BLOCKSIZE];
          __shared__ T shared_val[BLOCKSIZE];

          // loop over blocks that are subject for segmented reduction
          if (true) {
            constexpr int i = 0;
            // copy data to reduction buffers
            shared_row[tid_in_block] = (tid + i < nblocks) ? key_val[tid + i].key : -1;
            shared_val[tid_in_block] = (tid + i < nblocks) ? key_val[tid + i].value : static_cast<T>(0);

            __syncthreads();

            // do segmented block reduction
            usparse::primitives::reduce_by_key::block::segmented_blockreduce<I, T, BLOCKSIZE>(shared_row, shared_val);

            // Add reduced sum to y if valid
            const I row = shared_row[tid_in_block];

            if (tid_in_block == BLOCKSIZE - 1) {
              // the last thread in the block writes to temp buffer
              if (row != -1) {
                temp_data[blockIdx.x].key = row;
                temp_data[blockIdx.x].value = shared_val[tid_in_block];
              }
            } else {
              const I rowp1 = (tid_in_block < BLOCKSIZE - 1) ? shared_row[tid_in_block + 1] : -1;
              if (row != rowp1 && row >= 0) {
                y[row] = y[row] + shared_val[tid_in_block];
              }
            }
            __syncthreads();
          }
          return;
        }

        template <typename ROW, typename Y_TYPE, typename HIP_BLOCK_NUM_TYPE, int THREADS_PER_BLOCK>
        inline void flat_reduce_carry_out(hipStream_t stream, Y_TYPE *__restrict__ y,
                                          KV<ROW, Y_TYPE> *__restrict__ key_val, const HIP_BLOCK_NUM_TYPE HIP_BLOCKS) {
          if (HIP_BLOCKS < 2 * THREADS_PER_BLOCK) {
            final_segmented_loops_reduce_device<THREADS_PER_BLOCK, ROW, Y_TYPE, Y_TYPE>
                <<<1, THREADS_PER_BLOCK, 0, stream>>>(HIP_BLOCKS, key_val, y);
          } else {
            int NEW_HIP_BLOCKS = (HIP_BLOCKS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            KV<ROW, Y_TYPE> *temp_data = key_val + HIP_BLOCKS;
            segmented_loops_reduce_device<THREADS_PER_BLOCK, ROW, Y_TYPE, Y_TYPE>
                <<<NEW_HIP_BLOCKS, THREADS_PER_BLOCK, 0, stream>>>(HIP_BLOCKS, key_val, y, temp_data);
            hipStreamSynchronize(stream);
            flat_reduce_carry_out<ROW, Y_TYPE, HIP_BLOCK_NUM_TYPE, THREADS_PER_BLOCK>(stream, y, temp_data,
                                                                                      NEW_HIP_BLOCKS);
          }
          return;
        }
      } // namespace device
    } // namespace reduce_by_key
  } // namespace primitives
} // namespace usparse

#endif // USPARSE_KERNELS__USPARSE_REDUCE_BY_KEY_HPP
