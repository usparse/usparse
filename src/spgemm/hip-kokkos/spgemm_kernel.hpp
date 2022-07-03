#ifndef SPGEMM_KOKKOS_SPGEMM_KERNEL_HPP
#define SPGEMM_KOKKOS_SPGEMM_KERNEL_HPP

#include "hash_map.h"
#include "build_flag.h"
#include <stdio.h>

template<typename T, typename I, size_t BitsLen, size_t WF_SIZE>
__global__ void compress_matrix_kernel(const size_t m, const size_t n, const I* row_ptr, const I* col_idx_ptr, T* arr, const size_t nums_per_row) {
	// thread id in global
	const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
	// thread num in global
	const int global_thread_num = gridDim.x * blockDim.x;
	// each thread process a row
	for (int row = global_thread_id; row < m; row += global_thread_num) {
		const I row_idx_begin = row_ptr[row];
		const I row_idx_end = row_ptr[row + 1];
		for (int idx = row_idx_begin; idx < row_idx_end; idx++) {
			const I col = col_idx_ptr[idx];
                        const int bitset_idx = row * nums_per_row + col / BitsLen;
			arr[bitset_idx] |= 0x1 << (BitsLen - 1 - (col % BitsLen));
		}
	}
}

template<typename T, typename I, size_t BitsLen, size_t WF_SIZE>
__global__ void spgemm_symbolic_kernel(const size_t m, const size_t n, const I* row_ptr, const I* col_idx_ptr, const T* bitset_b, const size_t nums_per_row, I* row_ptr_c) {
	// thread id in global
	const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
	// thread num in global
	const int global_thread_num = gridDim.x * blockDim.x;
	// each thread process a row
	for (int row = global_thread_id; row < m; row += global_thread_num) {
		const I row_idx_begin = row_ptr[row];
		const I row_idx_end = row_ptr[row + 1];
		I row_count = 0;
		for (int offset = 0; offset < nums_per_row; ++offset) {
			T bits = 0;
			for (int idx = row_idx_begin; idx < row_idx_end; ++idx) {
				const I col = col_idx_ptr[idx];
				bits |= bitset_b[col * nums_per_row + offset];
			}
			while(bits != 0) {
				if (bits & 0x1) {
					row_count++;
				}
				bits = bits >> 1;
			}
		}
		row_ptr_c[row + 1] = row_count;
	}	
}

template<typename T, typename I, int N, size_t BLOCKS, size_t WF_SIZE>
__global__ void spgemm_numeric_kernel(const T alpha, const size_t m, const size_t n, const I* row_ptr_a, const I* col_idx_ptr_a, const T* values_ptr_a, 
																			const I* row_ptr_b, const I* col_idx_ptr_b, const T* values_ptr_b,
																			const I* row_ptr_c, int* begins, int* nexts, I* ids, T* values) {
	// thread id in global
	const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
	// thread num in global
	const int global_thread_num = gridDim.x * blockDim.x;
        // wf id in global
        const int global_wf_id = global_thread_id / WF_SIZE;
        // wf num in global
        const int global_wf_num = global_thread_num / WF_SIZE;
        // wf id in block
        const int block_wf_id = threadIdx.x / WF_SIZE; 
        // thread id in wf
        const int wf_thread_id = global_thread_id % WF_SIZE;
        // thread num in wf
        const int wf_thread_num = WF_SIZE;
        // hashmap in LDS
        __shared__ int shared_begins[HashMap<T, I>::Capacity];
        __shared__ int shared_nexts[WF_SIZE * N];
        __shared__ I shared_ids[WF_SIZE * N];
        __shared__ T shared_values[WF_SIZE * N];
	// each wf process one row
	__shared__ int shared_nexts_offset[BLOCKS/WF_SIZE];

	for (int row_a = global_wf_id; row_a < m; row_a += global_wf_num) {
		const I row_idx_begin = row_ptr_a[row_a];
		const I row_idx_end = row_ptr_a[row_a + 1];
		const size_t begins_start = HashMap<T,I>::Capacity * row_a;
		const size_t nexts_start = row_ptr_c[row_a];
                const size_t nexts_end = row_ptr_c[row_a + 1];
                if (wf_thread_id == 0) {
		    shared_nexts_offset[block_wf_id] = 0;
                }
		for (int idx_a = row_idx_begin; idx_a < row_idx_end; ++idx_a) {
			const I col_a = col_idx_ptr_a[idx_a];
			const I row_b = col_a;
			const I idx_begin_b = row_ptr_b[row_b];
			const I idx_end_b = row_ptr_b[row_b + 1];
			for (int idx_b = idx_begin_b + wf_thread_id; idx_b < idx_end_b; idx_b += wf_thread_num) {
				const T tmp_val = values_ptr_a[idx_a] * values_ptr_b[idx_b];
				const I col_b = col_idx_ptr_b[idx_b];
				const size_t hash_val = HASH(T, I, col_b);
                                hashmap_wavefront_insert(begins, nexts, ids, values, shared_nexts_offset, block_wf_id, begins_start, nexts_start, col_b, tmp_val);	
			}
		}
                for(int values_idx = nexts_start + wf_thread_id; values_idx < nexts_end; values_idx += wf_thread_num) {
                        values[values_idx] *= alpha;
                }
	}	
}

#endif //SPGEMM_KOKKOS_SPGEMM_KERNEL_HPP
