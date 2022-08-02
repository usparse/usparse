#ifndef SPGEMM_KOKKOS_HASH_MAP_H
#define SPGEMM_KOKKOS_HASH_MAP_H

#include "error_macro.hpp"
#include <hip/hip_runtime.h>

// tips: array in hashmap allocate in device
template <typename T, typename I> struct HashMap {
  using dtype = T;
  using csrIdxType = I;
  static constexpr int M = 10;
  static constexpr size_t Capacity = 0x1 << M;
  static constexpr size_t HashMask = Capacity - 1;
  static constexpr int NullValue = -1;
  static constexpr int PlaceHolderValue = -2;

  csrIdxType *ids;
  dtype *values;
  int *begins;
  int *nexts;

  HashMap(csrIdxType *col_idx_ptr, dtype *values_ptr, size_t rows, size_t nnz) : ids(col_idx_ptr), values(values_ptr) {
    KOKKOS_HIP_CHECK(hipMalloc((void **)&begins, sizeof(int) * HashMap::Capacity * rows))
    KOKKOS_HIP_CHECK(hipMemset(begins, HashMap::NullValue, sizeof(int) * HashMap::Capacity * rows))
    KOKKOS_HIP_CHECK(hipMalloc((void **)&nexts, sizeof(int) * nnz))
    KOKKOS_HIP_CHECK(hipMemset(nexts, HashMap::NullValue, sizeof(int) * nnz))
  }

  ~HashMap() {
    if (begins != nullptr) {
      KOKKOS_HIP_CHECK(hipFree(begins))
    }
    if (nexts != nullptr) {
      KOKKOS_HIP_CHECK(hipFree(nexts))
    }
  }
};

#define HASH(T, I, n, mask) (n & mask)

// one thread simultaneously insert to hashmap
template <typename T, typename I, size_t HashMask>
__device__ __forceinline__ void hashmap_thread_insert(int *begins, int *nexts, I *ids, T *values,
                                                      const size_t begins_start, const size_t nexts_start,
                                                      size_t &nexts_offset, const I insert_id, const T insert_value) {
  const size_t key = HASH(T, I, insert_id, HashMask);
  const int begin_val = begins[begins_start + key];
  if (begin_val == HashMap<T, I>::NullValue) {
    begins[begins_start + key] = nexts_start + nexts_offset;
    ids[nexts_start + nexts_offset] = insert_id;
    values[nexts_start + nexts_offset] = insert_value;
    ++nexts_offset;
  } else {
    int next_idx = begin_val;
    while (nexts[next_idx] != HashMap<T, I>::NullValue) {
      if (ids[next_idx] == insert_id) {
        values[next_idx] += insert_value;
        break;
      }
      next_idx = nexts[next_idx];
    }
    if (nexts[next_idx] == HashMap<T, I>::NullValue) {
      if (ids[next_idx] == insert_id) {
        values[next_idx] += insert_value;
      } else {
        nexts[next_idx] = nexts_start + nexts_offset;
        next_idx = nexts_start + nexts_offset;
        nexts[next_idx] = HashMap<T, I>::NullValue;
        ids[next_idx] = insert_id;
        values[next_idx] = insert_value;
        ++nexts_offset;
      }
    }
  }
}

// threads in wavefront simultaneously insert to hashmap
template <typename T, typename I, size_t HashMask>
__device__ __forceinline__ void
hashmap_wavefront_insert(int *begins, int *nexts, I *ids, T *values, int *nexts_offsets, int offsets_index,
                         const size_t begins_start, const size_t nexts_start, const I insert_id, const T insert_value) {
  const size_t hash_val = HASH(T, I, insert_id, HashMask);
  volatile int begin_val = begins[begins_start + hash_val];
  int nexts_offset = nexts_offsets[offsets_index];
  // 1. if key not in hashmap, insert (id, val) to map
  if (begin_val == HashMap<T, I>::NullValue) {
    int old = atomicCAS(begins + begins_start + hash_val, HashMap<T, I>::NullValue, HashMap<T, I>::PlaceHolderValue);
    begin_val = begins[begins_start + hash_val];
    while (begin_val == HashMap<T, I>::PlaceHolderValue) {
      if (old == HashMap<T, I>::NullValue) {
        // ++nexts_offset;
        nexts_offset = atomicAdd(nexts_offsets + offsets_index, 1);
        atomicExch(begins + begins_start + hash_val, nexts_start + nexts_offset);
        begin_val = nexts_start + nexts_offset;
        ids[nexts_start + nexts_offset] = insert_id;
        values[nexts_start + nexts_offset] = insert_value;
      } else {
        // notice: Do not merge the above branches
        begin_val = begins[begins_start + hash_val];
      }
    }
    if (old == HashMap<T, I>::NullValue) {
      return;
    }
  }
  // 2. if key in hashmap, check if hashmap have same id
  // 3. if have same id, add val in origin val
  // 4. if have not same id, insert after Null node
  int next_idx = begin_val;
  while (nexts[next_idx] != HashMap<T, I>::NullValue) {
    if (ids[next_idx] == insert_id) {
      values[next_idx] += insert_value;
      break;
    }
    next_idx = nexts[next_idx];
  }
  if (nexts[next_idx] == HashMap<T, I>::NullValue) {
    if (ids[next_idx] == insert_id) {
      values[next_idx] += insert_value;
    } else {
      // col index of b in a row can't be same
      int old_offset = atomicAdd(nexts_offsets + offsets_index, 1);
      while (true) {
        int old = atomicCAS(nexts + next_idx, HashMap<T, I>::NullValue, nexts_start + old_offset);
        if (old == HashMap<T, I>::NullValue) {
          next_idx = nexts_start + old_offset;
          ids[next_idx] = insert_id;
          values[next_idx] = insert_value;
          break;
        } else {
          next_idx = nexts[next_idx];
        }
      }
    }
  }
}
#endif // SPGEMM_KOKKOS_HASH_MAP_H
