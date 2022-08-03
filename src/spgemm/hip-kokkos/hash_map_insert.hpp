#ifndef SPGEMM_KOKKOS_HASH_MAP_KERNEL_HPP
#define SPGEMM_KOKKOS_HASH_MAP_KERNEL_HPP

#include <hip/hip_runtime.h>

#include "hash_map.h"

// one thread simultaneously insert to hashmap
template <typename T, typename I, size_t HashMask>
__device__ __forceinline__ void hashmap_thread_insert_add(int *begins, int *nexts, I *ids, T *values, const I insert_id,
                                                          const T insert_value, int *use_size) {
  const size_t hash_val = HASH(T, I, insert_id, HashMask);
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
__device__ __forceinline__ void hashmap_thread_insert_or(int *begins, int *nexts, I *ids, T *values, const I insert_id,
                                                         const T insert_value, int *use_size) {
  const size_t hash_val = HASH(T, I, insert_id, HashMask);
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
__device__ __forceinline__ void Ihashmap_vector_insert_add(int *begins, int *nexts, I *ids, T *values,
                                                           const I insert_id, const T insert_value, int *use_size) {
  const size_t hash_val = HASH(T, I, insert_id, HashMask);
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
__device__ __forceinline__ void Ihashmap_vector_insert_or(int *begins, int *nexts, I *ids, T *values, const I insert_id,
                                                          const T insert_value, int *use_size) {
  const size_t hash_val = HASH(T, I, insert_id, HashMask);
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
