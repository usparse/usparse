#ifndef SPGEMM_HIP_KOKKOS_BITSET_H
#define SPGEMM_HIP_KOKKOS_BITSET_H

struct BitSet {
  using BitSetEleType = unsigned int;
  static constexpr size_t BitsPerByte = 8;
  static constexpr size_t BitsLen = sizeof(BitSetEleType) * BitsPerByte;
  const size_t rows;
  const size_t cols;
  BitSetEleType *arr;
  BitSet(size_t r, size_t c, BitSetEleType *arr) : rows(r), cols(c), arr(arr) {}
};

#endif // SPGEMM_HIP_KOKKOS_BITSET_H