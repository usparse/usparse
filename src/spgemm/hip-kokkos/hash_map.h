#ifndef SPGEMM_KOKKOS_HASH_MAP_H
#define SPGEMM_KOKKOS_HASH_MAP_H

#include "error_macro.hpp"
#include <cstddef>
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
__device__ __forceinline__ void hashmap_wavefront_insert_add(int *begins, int *nexts, I *ids, T *values,
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
__device__ __forceinline__ void hashmap_wavefront_insert_or(int *begins, int *nexts, I *ids, T *values,
                                                            const I insert_id, const T insert_value, int *use_size) {
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
#endif // SPGEMM_KOKKOS_HASH_MAP_H
