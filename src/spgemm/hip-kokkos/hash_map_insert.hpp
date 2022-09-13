#ifndef SPGEMM_KOKKOS_HASH_MAP_KERNEL_HPP
#define SPGEMM_KOKKOS_HASH_MAP_KERNEL_HPP

#include <hip/hip_runtime.h>

#include "hash.h"
#include "hash_map.h"

// ** insert to hashset (open addressing) **
template <typename T, typename I, size_t HashMask>
__device__ __forceinline__ void open_address_hashset_vector_insert(I *ids, const I insert_id) {
  size_t hash_val = HASH(insert_id, HashMask);

  while (true) {
    if (ids[hash_val] == insert_id) {
      break;
    } else if (ids[hash_val] == HashMap<T, I>::NullValue) {
      if (atomicCAS(ids + hash_val, HashMap<T, I>::NullValue, insert_id) == HashMap<T, I>::NullValue) {
        break;
      }
    } else {
      hash_val = HASH(hash_val + 1, HashMask);
    }
  }
}

template <typename T, typename I, size_t HashMask>
__device__ __forceinline__ void open_address_hashset_vector_insert(I *ids, const I insert_id, int *use_count) {
  size_t hash_val = HASH(insert_id, HashMask);

  while (true) {
    if (ids[hash_val] == insert_id) {
      break;
    } else if (ids[hash_val] == HashMap<T, I>::NullValue) {
      if (atomicCAS(ids + hash_val, HashMap<T, I>::NullValue, insert_id) == HashMap<T, I>::NullValue) {
        atomicAdd(use_count, 1);
        break;
      }
    } else {
      hash_val = HASH(hash_val + 1, HashMask);
    }
  }
}

// ** insert to hashmap (open addressing) **
template <typename T, typename I, size_t HashMask>
__device__ __forceinline__ void open_address_hashmap_vector_insert_or(I *ids, T *values, const I insert_id,
                                                                      const T insert_value) {
  size_t hash_val = HASH(insert_id, HashMask);

  while (true) {
    if (ids[hash_val] == insert_id) {
      atomicOr(values + hash_val, insert_value);
      break;
    } else if (ids[hash_val] == HashMap<T, I>::NullValue) {
      if (atomicCAS(ids + hash_val, HashMap<T, I>::NullValue, insert_id) == HashMap<T, I>::NullValue) {
        atomicOr(values + hash_val, insert_value);
        break;
      }
    } else {
      hash_val = HASH(hash_val + 1, HashMask);
    }
  }
}

template <typename T, typename I, size_t HashMask>
__device__ __forceinline__ void open_address_hashmap_vector_insert_add(I *ids, T *values, const I insert_id,
                                                                       const T insert_value) {
  size_t hash_val = HASH(insert_id, HashMask);

  while (true) {
    if (ids[hash_val] == insert_id) {
      atomicAdd(values + hash_val, insert_value);
      break;
    } else if (ids[hash_val] == HashMap<T, I>::NullValue) {
      if (atomicCAS(ids + hash_val, HashMap<T, I>::NullValue, insert_id) == HashMap<T, I>::NullValue) {
        atomicAdd(values + hash_val, insert_value);
        break;
      }
    } else {
      hash_val = HASH(hash_val + 1, HashMask);
    }
  }
}

// ** insert to hashmap (linear chaining) **
template <typename T, typename I, size_t HashMask>
__device__ __forceinline__ void linear_chain_hashmap_thread_insert_add(int *begins, int *nexts, I *ids, T *values,
                                                                       const I insert_id, const T insert_value,
                                                                       int *use_size) {
  const size_t hash_val = HASH(insert_id, HashMask);
  int i;
  i = begins[hash_val];
  for (; i != HashMap<T, I>::NullValue; i = nexts[i]) {
    if (ids[i] == insert_id) {
      values[i] += insert_value;
      return;
    }
  }
  size_t index = (*use_size)++;
  nexts[index] = begins[hash_val];

  begins[hash_val] = index;
  ids[index] = insert_id;
  values[index] = insert_value;
}

template <typename T, typename I, size_t HashMask>
__device__ __forceinline__ void linear_chain_hashmap_thread_insert_or(int *begins, int *nexts, I *ids, T *values,
                                                                      const I insert_id, const T insert_value,
                                                                      int *use_size) {
  const size_t hash_val = HASH(insert_id, HashMask);
  int i;
  i = begins[hash_val];
  for (; i != HashMap<T, I>::NullValue; i = nexts[i]) {
    if (ids[i] == insert_id) {
      values[i] |= insert_value;
      return;
    }
  }
  size_t index = (*use_size)++;
  nexts[index] = begins[hash_val];

  begins[hash_val] = index;
  ids[index] = insert_id;
  values[index] = insert_value;
}

// threads in wavefront simultaneously insert to hashmap
template <typename T, typename I, size_t HashMask>
__device__ __forceinline__ void linear_chain_hashmap_vector_insert_add(int *begins, int *nexts, I *ids, T *values,
                                                                       const I insert_id, const T insert_value,
                                                                       int *use_size) {
  const size_t hash_val = HASH(insert_id, HashMask);
  int i;
  i = begins[hash_val];
  for (; i != HashMap<T, I>::NullValue; i = nexts[i]) {
    if (ids[i] == insert_id) {
      values[i] += insert_value;
      return;
    }
  }
  int write_index = atomicAdd(use_size, 1);
  ids[write_index] = insert_id;
  values[write_index] = insert_value;
  int hash_begin = atomicExch(begins + hash_val, write_index);
  nexts[write_index] = hash_begin;
}

template <typename T, typename I, size_t HashMask>
__device__ __forceinline__ void linear_chain_hashmap_vector_insert_or(int *begins, int *nexts, I *ids, T *values,
                                                                      const I insert_id, const T insert_value,
                                                                      int *use_size) {
  const size_t hash_val = HASH(insert_id, HashMask);
  int i;
  i = begins[hash_val];
  for (; i != HashMap<T, I>::NullValue; i = nexts[i]) {
    if (ids[i] == insert_id) {
      values[i] |= insert_value;
      return;
    }
  }
  int write_index = atomicAdd(use_size, 1);
  ids[write_index] = insert_id;
  values[write_index] = insert_value;
  int hash_begin = atomicExch(begins + hash_val, write_index);
  nexts[write_index] = hash_begin;
}

#endif // SPGEMM_KOKKOS_HASH_MAP_KERNEL_HPP
