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

template<typename T, typename I, size_t BLOCKS, size_t WF_SIZE>
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
				T tmp_val = values_ptr_a[idx_a] * values_ptr_b[idx_b];
				const I col_b = col_idx_ptr_b[idx_b];
				const size_t hash_val = HASH(T, I, col_b);
#ifdef DEV_MODE
                                printf("[%d][%d][%d] (key:%d, value:%f, hash:%zu)\n", row_a, global_wf_id, wf_thread_id, col_b, tmp_val, hash_val);
#endif
                                volatile int begin_val = begins[begins_start + hash_val];
                                int nexts_offset = shared_nexts_offset[block_wf_id];
                                // 1. if key not in hashmap, insert (id, val) to map
				if (begin_val == HashMap<T,I>::NullValue) {
                                        int old = atomicCAS(begins + begins_start + hash_val, HashMap<T,I>::NullValue, HashMap<T,I>::PlaceHolderValue); 
#ifdef DEV_MODE
                                            printf("[%d][%d][%d](old:%d)\n", row_a, global_wf_id, wf_thread_id, old);
#endif
                                        begin_val = begins[begins_start + hash_val];
                                        while (begin_val == HashMap<T,I>::PlaceHolderValue) {
                                            if (old == HashMap<T,I>::NullValue) {
				 	        // ++nexts_offset;
				 	        nexts_offset = atomicAdd(shared_nexts_offset + block_wf_id, 1);
					        begins[begins_start + hash_val] = nexts_start + nexts_offset; 
                                                begin_val = nexts_start + nexts_offset;
					        ids[nexts_start + nexts_offset] = col_b;
					        values[nexts_start + nexts_offset] = tmp_val;
#ifdef DEV_MODE
                                                printf("[%d][%d][%d]insert head node(hash:%zu) %zu->%lu\n", row_a, global_wf_id, wf_thread_id, hash_val, hash_val, nexts_start + nexts_offset);
#endif
                                            } else {
                                                // notice: Do not merge the above branches
                                                begin_val = begins[begins_start + hash_val];
                                            }
                                            // begin_val = begins[begins_start + hash_val];
                                       }
                                       if (old == HashMap<T,I>::NullValue) {
                                            continue;
                                       } else {
#ifdef DEV_MODE
                                            printf("[%d][%d][%d]loop(begin_val:%d)\n", row_a, global_wf_id, wf_thread_id, begin_val);
#endif
                                       }
				}
                                {
                                        // 2. if key in hashmap, check if hashmap have same id
                                        // 3. if have same id, add val in origin val
                                        // 4. if have not same id, insert after Null node 
				    	int next_idx = begin_val;
					while(nexts[next_idx] != HashMap<T,I>::NullValue) {
#ifdef DEV_MODE
                                                printf("[%d][%d][%d]find next node(hash_val:%zu) %d->%lu\n", row_a, global_wf_id, wf_thread_id, hash_val, next_idx, nexts[next_idx]);
#endif

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
                                                        // col index of b in a row can't be same
                                                        int old_offset = atomicAdd(shared_nexts_offset + block_wf_id, 1);
                                                        while(true) {
                                                            int old = atomicCAS(nexts + next_idx, HashMap<T,I>::NullValue, nexts_start + old_offset);
                                                            if (old == HashMap<T,I>::NullValue) {
#ifdef DEV_MODE
                                                                printf("[%d][%d][%d]insert new node(hash:%zu) %d->%lu\n", row_a, global_wf_id, wf_thread_id, hash_val, next_idx, nexts_start + old_offset);
#endif
							        // nexts[next_idx] = nexts_start + shared_nexts_offset[block_wf_id];
							        next_idx = nexts_start + old_offset;
							        // nexts[next_idx] = HashMap<T,I>::NullValue;
							        ids[next_idx] = col_b;
							        values[next_idx] = tmp_val;
                                                                break;
                                                            } else {
                                                                next_idx = nexts[next_idx];
#ifdef DEV_MODE
                                                                printf("[%d][%d][%d]goto next(hash:%zu) node %d\n", row_a, global_wf_id, wf_thread_id, hash_val, next_idx);
#endif
                                                            }
                                                        } 
						}
					}	
				}
			}
		}
                for(int values_idx = nexts_start + wf_thread_id; values_idx < nexts_end; values_idx += wf_thread_num) {
                        values[values_idx] *= alpha;
                }
	}	
}

#endif //SPGEMM_KOKKOS_SPGEMM_KERNEL_HPP
