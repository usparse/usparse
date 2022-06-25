#ifndef SPGEMM_KOKKOS_SPGEMM_KERNEL_HPP
#define SPGEMM_KOKKOS_SPGEMM_KERNEL_HPP

#include "hash_map.h"

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

template<typename T, typename I, size_t WF_SIZE>
__global__ void spgemm_numeric_kernel(const T alpha, const size_t m, const size_t n, const I* row_ptr_a, const I* col_idx_ptr_a, const T* values_ptr_a, 
																			const I* row_ptr_b, const I* col_idx_ptr_b, const T* values_ptr_b,
																			const I* row_ptr_c, int* begins, int* nexts, I* ids, T* values) {
	// thread id in global
	const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
	// thread num in global
	const int global_thread_num = gridDim.x * blockDim.x;
	// each thread process a row
	for (int row_a = global_thread_id; row_a < m; row_a += global_thread_num) {
		const I row_idx_begin = row_ptr_a[row_a];
		const I row_idx_end = row_ptr_a[row_a + 1];
		const size_t begins_start = HashMap<T,I>::Capacity * row_a;
		const size_t nexts_start = row_ptr_c[row_a];
		int nexts_offset = 0;
		for (int idx_a = row_idx_begin; idx_a < row_idx_end; ++idx_a) {
			const I col_a = col_idx_ptr_a[idx_a];
			const I row_b = col_a;
			const I idx_begin_b = row_ptr_b[row_b];
			const I idx_end_b = row_ptr_b[row_b + 1];
			for (int idx_b = idx_begin_b; idx_b < idx_end_b; ++idx_b) {
				T tmp_val = alpha * values_ptr_a[idx_a] * values_ptr_b[idx_b];
				const I col_b = col_idx_ptr_b[idx_b];
				const size_t key = HASH(T, I, col_b);
				const int begin_val = begins[begins_start + key];
				if (begin_val == HashMap<T,I>::NullValue) {
					begins[begins_start + key] = nexts_start + nexts_offset;
					ids[nexts_start + nexts_offset] = col_b;
					values[nexts_start + nexts_offset] = tmp_val;
				 	++nexts_offset;
				} else {
					int next_idx = begin_val;
					while(nexts[next_idx] != HashMap<T,I>::NullValue) {
						if (ids[next_idx] == col_b) {
							values[next_idx] += tmp_val;
							break;
						}
						next_idx = nexts[next_idx];
					}
					if (nexts[next_idx] == HashMap<T,I>::NullValue) {
						if (ids[next_idx] == col_b) {
							values[next_idx] += tmp_val;
						} else {
							nexts[next_idx] = nexts_start + nexts_offset;
							next_idx = nexts_start + nexts_offset;
							nexts[next_idx] = HashMap<T,I>::NullValue;
							ids[next_idx] = col_b;
							values[next_idx] = tmp_val;
							++nexts_offset;
						}
					}	
				}
			}
		}
	}	
}

#endif //SPGEMM_KOKKOS_SPGEMM_KERNEL_HPP
