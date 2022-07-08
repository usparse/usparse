#ifndef SPGEMM_KOKKOS_SPGEMM_CORE_HPP
#define SPGEMM_KOKKOS_SPGEMM_CORE_HPP

#include <iostream>
#include <bitset>
#include <hip/hip_runtime.h>
#include <api/sort_hip_api.h>
#include "spgemm_core.h"
#include "error_macro.hpp"
#include "spgemm_kernel.hpp"
#include "utils_kernel.hpp"
#include "hash_map.h"
#include "build_flag.h"

template <typename T, typename I>
CsrMatrix<T, I> spgemm_core(const T alpha, const CsrMatrix<T, I>& matrix_a, const CsrMatrix<T, I>& matrix_b, I* row_ptr_c) {
  // Compress matrix B
  BitSet bitset_matrix_b = compress_matrix(matrix_b);
  // Symbolic Phase
  size_t nnz_c = 0;
  spgemm_symbolic<T, I>(matrix_a, bitset_matrix_b, row_ptr_c, &nnz_c);
  KOKKOS_HIP_CHECK( hipFree(bitset_matrix_b.arr) )
  // Numeric Phase 
  CsrMatrix<T, I> matrix_c(matrix_a.rows, matrix_b.cols, nnz_c, row_ptr_c, nullptr, nullptr);
  KOKKOS_HIP_CHECK( hipMalloc((void**) &matrix_c.col_idx_ptr, nnz_c * sizeof(I)) )
  KOKKOS_HIP_CHECK( hipMalloc((void**) &matrix_c.values_ptr, nnz_c * sizeof(T)) )
  spgemm_numeric(alpha, matrix_a, matrix_b, matrix_c);
  // Sort Matrix
  csr_matrix_sort<T, I>(matrix_c);
  // Kernels need to implement 
  return matrix_c;
}

template <typename T, typename I>
BitSet compress_matrix(const CsrMatrix<T, I>& csr_matrix) {
  const size_t nums_per_row = csr_matrix.cols / BitSet::BitsLen + (csr_matrix.cols % BitSet::BitsLen == 0 ? 0 : 1); 
  BitSet bitset(csr_matrix.rows, nums_per_row, nullptr);
  const size_t arr_len = bitset.rows * bitset.cols * sizeof(BitSet::BitSetEleType);
  KOKKOS_HIP_CHECK( hipMalloc((void**) &bitset.arr, arr_len) )
  KOKKOS_HIP_CHECK( hipMemset(bitset.arr, 0, arr_len) )
  (compress_matrix_kernel<BitSet::BitSetEleType, I, BitSet::BitsLen, 64>)<<<512, 256>>>(csr_matrix.rows, csr_matrix.cols, csr_matrix.row_ptr, csr_matrix.col_idx_ptr, bitset.arr, bitset.cols);

#ifdef DEV_MODE
  // print col index
  I* host_col_idx = new I[csr_matrix.nnz];
  KOKKOS_HIP_CHECK( hipMemcpy(host_col_idx, csr_matrix.col_idx_ptr, csr_matrix.nnz * sizeof(I), hipMemcpyDeviceToHost) )
  I* host_row_ptr = new I[csr_matrix.rows + 1];
  KOKKOS_HIP_CHECK( hipMemcpy(host_row_ptr, csr_matrix.row_ptr, (csr_matrix.rows + 1) * sizeof(I), hipMemcpyDeviceToHost))
  std::cout << "col index: " << std::endl;
  for (int i = 0; i < csr_matrix.rows; ++i) {
    const I begin = host_row_ptr[i];
    const I end = host_row_ptr[i + 1];
    for (int j = begin; j < end; ++j) {
      std::cout << host_col_idx[j];
      if (j != end - 1) {
        std::cout << ", ";
      }
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  delete[] host_row_ptr;
  delete[] host_col_idx;
  // print bitset 
  BitSet::BitSetEleType* host_arr = new BitSet::BitSetEleType[bitset.rows * bitset.cols];
  KOKKOS_HIP_CHECK( hipMemcpy(host_arr, bitset.arr, arr_len, hipMemcpyDeviceToHost) )
  std::cout << "bitset result: " << std::endl;
  for (int i = 0; i < bitset.rows; ++i) {
    for (int j = 0; j < bitset.cols; ++j) {
      std::cout << std::bitset<BitSet::BitsLen>(host_arr[i * bitset.cols + j]);
    }
    std::cout << std::endl;
  }
  delete[] host_arr;
#endif
  return bitset;
}

template <typename T, typename I>
void spgemm_symbolic(const CsrMatrix<T, I>& matrix_a, const BitSet& bitset_b, I* row_ptr_c, size_t* nnz_c) {
  (spgemm_symbolic_kernel<BitSet::BitSetEleType, I, BitSet::BitsLen, 64>)<<<512, 256>>>(matrix_a.rows, matrix_a.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, bitset_b.arr, bitset_b.cols, row_ptr_c);
  (continuous_acc_kernel<I, 512, 64>)<<<1, 256>>>(matrix_a.rows + 1, row_ptr_c); 
  // set nnz of matrix c
  KOKKOS_HIP_CHECK( hipMemcpy(nnz_c, &row_ptr_c[matrix_a.rows], sizeof(I), hipMemcpyDeviceToHost) )

#ifdef DEV_MODE
  I* host_row_ptr_c = new I[matrix_a.rows + 1];
  KOKKOS_HIP_CHECK( hipMemcpy(host_row_ptr_c, row_ptr_c, (matrix_a.rows + 1) * sizeof(I), hipMemcpyDeviceToHost) )
  std::cout << "row result: " << std::endl;
  for (int i = 0; i < matrix_a.rows + 1; ++i) {
    std::cout << i << "\t:" << host_row_ptr_c[i] << std::endl;
  }
  delete[] host_row_ptr_c;
#endif
}

template <typename T, typename I>
void spgemm_numeric(const T alpha, const CsrMatrix<T, I>& matrix_a, const CsrMatrix<T, I>& matrix_b, CsrMatrix<T, I>& matrix_c) {
  HashMap<T,I> hash_map(matrix_c.col_idx_ptr, matrix_c.values_ptr, matrix_c.rows, matrix_c.nnz);
  constexpr size_t N = 8;
  constexpr size_t LDS_MAP_CAP = 1 << 5;
  constexpr size_t LDS_HASH_MASK = LDS_MAP_CAP - 1;
  constexpr size_t GLOBAL_BLOCK_NUM = 512;
  constexpr size_t BLOCK_THREAD_NUM = 256;
  constexpr size_t WF_SIZE  = 64;
  (spgemm_numeric_kernel<T, I, N, LDS_MAP_CAP, LDS_HASH_MASK, BLOCK_THREAD_NUM, WF_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(alpha, matrix_a.rows, matrix_a.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
                                      matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, 
                                      matrix_c.row_ptr, hash_map.begins, hash_map.nexts, hash_map.ids, hash_map.values);

#ifdef DEV_MODE
  I* host_row_ptr_c = new I[matrix_c.rows + 1];
  KOKKOS_HIP_CHECK( hipMemcpy(host_row_ptr_c, matrix_c.row_ptr, (matrix_c.rows + 1) * sizeof(I), hipMemcpyDeviceToHost) )

  I* host_col_idx_ptr_c = new I[matrix_c.nnz];
  T* host_values_ptr_c = new T[matrix_c.nnz];
  KOKKOS_HIP_CHECK( hipMemcpy(host_col_idx_ptr_c, matrix_c.col_idx_ptr, (matrix_c.nnz) * sizeof(I), hipMemcpyDeviceToHost) )
  KOKKOS_HIP_CHECK( hipMemcpy(host_values_ptr_c, matrix_c.values_ptr, (matrix_c.nnz) * sizeof(T), hipMemcpyDeviceToHost) )
  std::cout << "(col, val) result: " << std::endl;
  for (int i = 0; i < matrix_c.rows; ++i) {
    std::cout << i << "\t:";
    for (int idx = host_row_ptr_c[i]; idx < host_row_ptr_c[i + 1]; ++idx) {
      std::cout << "[" << host_col_idx_ptr_c[idx] << ", " << host_values_ptr_c[idx] << "]";
      if (idx != host_row_ptr_c[i + 1] - 1) {
        std::cout << ", ";
      }
    }
    std::cout << std::endl;
  }
  delete[] host_values_ptr_c;
  delete[] host_col_idx_ptr_c;
  delete[] host_row_ptr_c;
#endif
}

template <typename T, typename I>
void csr_matrix_sort(CsrMatrix<T, I>& matrix) {
  csr_sort_hip<T, I>(matrix.rows, matrix.nnz, matrix.row_ptr, matrix.col_idx_ptr, matrix.values_ptr); 
}

#endif //SPGEMM_KOKKOS_SPGEMM_CORE_HPP
