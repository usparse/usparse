#ifndef SPGEMM_KOKKOS_SPGEMM_CORE_HPP
#define SPGEMM_KOKKOS_SPGEMM_CORE_HPP

#include "build_flag.h"
#include "compress_matrix.h"
#include "dense_to_sparse.hpp"
#include "error_macro.hpp"
#include "hash_map.h"
#include "scan.h"
#include "spgemm_core.h"
#include "spgemm_kernel.hpp"
#include "timer.h"
#include <api/sort_hip_api.h>
#include <bitset>
#include <cstddef>
#include <hip/hip_runtime.h>
#include <iostream>
#include <rocprim/rocprim.hpp>
#include <rocsparse.h>

template <typename T, typename I>
CsrMatrix<T, I> spgemm_core(const T alpha, const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b,
                            I *row_ptr_c) {
#ifndef TIME_MODE
  // Compress matrix B
  CompressMatrix<I> compress_matrix_b = compress_matrix(matrix_b);
  // Symbolic Phase
  size_t nnz_c = 0;
  spgemm_symbolic<T, I>(matrix_a, compress_matrix_b, row_ptr_c, &nnz_c);
  compress_matrix_b.free();
  // Numeric Phase
  CsrMatrix<T, I> matrix_c(matrix_a.rows, matrix_b.cols, nnz_c, row_ptr_c, nullptr, nullptr);
  KOKKOS_HIP_CHECK(hipMalloc((void **)&matrix_c.col_idx_ptr, nnz_c * sizeof(I)))
  KOKKOS_HIP_CHECK(hipMalloc((void **)&matrix_c.values_ptr, nnz_c * sizeof(T)))
  spgemm_numeric(alpha, matrix_a, matrix_b, matrix_c);
  // Sort Matrix
  csr_matrix_sort<T, I>(matrix_c);
#endif
#ifdef TIME_MODE
  Timer timer;
  hipDeviceSynchronize();
  // Compress matrix B
  timer.start();
  CompressMatrix<I> compress_matrix_b = compress_matrix(matrix_b);
  hipDeviceSynchronize();
  timer.stop();
  std::cout << "Compress phase: " << timer.time_use.count() << "(us)" << std::endl;
  // Symbolic Phase
  timer.start();
  size_t nnz_c = 0;
  spgemm_symbolic<T, I>(matrix_a, compress_matrix_b, row_ptr_c, &nnz_c);
  compress_matrix_b.free();
  hipDeviceSynchronize();
  timer.stop();
  std::cout << "Synbolic phase: " << timer.time_use.count() << "(us)" << std::endl;
  // Numeric Phase
  timer.start();
  CsrMatrix<T, I> matrix_c(matrix_a.rows, matrix_b.cols, nnz_c, row_ptr_c, nullptr, nullptr);
  KOKKOS_HIP_CHECK(hipMalloc((void **)&matrix_c.col_idx_ptr, nnz_c * sizeof(I)))
  KOKKOS_HIP_CHECK(hipMalloc((void **)&matrix_c.values_ptr, nnz_c * sizeof(T)))
  spgemm_numeric(alpha, matrix_a, matrix_b, matrix_c);
  hipDeviceSynchronize();
  timer.stop();
  std::cout << "Numeric phase: " << timer.time_use.count() << "(us)" << std::endl;
  // Sort Matrix
  timer.start();
  csr_matrix_sort<T, I>(matrix_c);
  hipDeviceSynchronize();
  timer.stop();
  std::cout << "Sort phase: " << timer.time_use.count() << "(us)" << std::endl;
#endif
  // Kernels need to implement
  return matrix_c;
}

template <typename T, typename I> CompressMatrix<I> compress_matrix(const CsrMatrix<T, I> &csr_matrix) {
  CompressMatrix<I> c_matrix(csr_matrix.rows, csr_matrix.cols, csr_matrix.nnz, csr_matrix.row_ptr);
  (compress_matrix_kernel<I, CompressMatrix<I>::BitsLen, 32, 64>)<<<512, 256>>>(
      csr_matrix.rows, csr_matrix.row_ptr, csr_matrix.col_idx_ptr, c_matrix.ids_ptr, c_matrix.col_idx_ptr,
      c_matrix.use_count);
#ifdef DEV_MODE
  // print col index
  I *host_col_idx = new I[csr_matrix.nnz];
  KOKKOS_HIP_CHECK(hipMemcpy(host_col_idx, csr_matrix.col_idx_ptr, csr_matrix.nnz * sizeof(I), hipMemcpyDeviceToHost))
  I *host_row_ptr = new I[csr_matrix.rows + 1];
  KOKKOS_HIP_CHECK(
      hipMemcpy(host_row_ptr, csr_matrix.row_ptr, (csr_matrix.rows + 1) * sizeof(I), hipMemcpyDeviceToHost))
  std::cout << "col index: " << std::endl;
  for (int i = 0; i < csr_matrix.rows; ++i) {
    const I begin = host_row_ptr[i];
    const I end = host_row_ptr[i + 1];
    std::cout << "<" << i << ">:";
    for (int j = begin; j < end; ++j) {
      std::cout << host_col_idx[j];
      if (j != end - 1) {
        std::cout << ", ";
      }
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  delete[] host_col_idx;
  // print bitset
  typename CompressMatrix<I>::IdType *host_ids = new typename CompressMatrix<I>::IdType[c_matrix.nnz];
  typename CompressMatrix<I>::BitSetEleType *host_values = new typename CompressMatrix<I>::BitSetEleType[c_matrix.nnz];
  int *host_use_count = new int[c_matrix.rows];
  KOKKOS_HIP_CHECK(hipMemcpy(host_ids, c_matrix.ids_ptr, c_matrix.nnz * sizeof(typename CompressMatrix<I>::IdType),
                             hipMemcpyDeviceToHost))
  KOKKOS_HIP_CHECK(hipMemcpy(host_values, c_matrix.col_idx_ptr,
                             c_matrix.nnz * sizeof(typename CompressMatrix<I>::BitSetEleType), hipMemcpyDeviceToHost))
  KOKKOS_HIP_CHECK(hipMemcpy(host_use_count, c_matrix.use_count, c_matrix.rows * sizeof(int), hipMemcpyDeviceToHost))
  std::cout << "bitset result: " << std::endl;
  for (int i = 0; i < csr_matrix.rows; ++i) {
    const I begin = host_row_ptr[i];
    const I end = host_row_ptr[i + 1];
    std::cout << "<" << i << ">,";
    std::cout << "<" << host_use_count[i] << ">:";
    for (int j = begin; j < end; ++j) {
      std::cout << "[" << host_ids[j] << ", " << std::bitset<CompressMatrix<I>::BitsLen>(host_values[j]) << "]->";
    }
    std::cout << std::endl;
  }
  delete[] host_use_count;
  delete[] host_row_ptr;
  delete[] host_values;
  delete[] host_ids;
#endif
  return c_matrix;
}

template <typename T, typename I>
void spgemm_symbolic(const CsrMatrix<T, I> &matrix_a, const CompressMatrix<I> &bitset_b, I *row_ptr_c, size_t *nnz_c) {
  using idType = typename CompressMatrix<I>::IdType;
  using bitType = typename CompressMatrix<I>::BitSetEleType;
  // calc size of trunk
  I *row_trunk_ptr = nullptr;
  KOKKOS_HIP_CHECK(hipMalloc((void **)(&row_trunk_ptr), (matrix_a.rows + 1) * sizeof(I)))
  KOKKOS_HIP_CHECK(hipMemset(row_trunk_ptr, 0, sizeof(I)))
  (trunk_nums_kernel<I, 32, 64>)<<<512, 256>>>(matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr,
                                               bitset_b.use_count, row_trunk_ptr);
  scan_in_place<I>(matrix_a.rows + 1, row_trunk_ptr);
  I trunk_num_c = 0;
  KOKKOS_HIP_CHECK(hipMemcpy(&trunk_num_c, row_trunk_ptr + matrix_a.rows, sizeof(I), hipMemcpyDeviceToHost))
  // alloc hashmap
  idType *map_ids = nullptr;
  bitType *map_values = nullptr;
  int *map_use_count = nullptr;
  KOKKOS_HIP_CHECK(hipMalloc((void **)(&map_ids), trunk_num_c * sizeof(idType)));
  KOKKOS_HIP_CHECK(hipMalloc((void **)(&map_values), trunk_num_c * sizeof(bitType)));
  KOKKOS_HIP_CHECK(hipMalloc((void **)(&map_use_count), matrix_a.rows * sizeof(int)));
  HashMap<bitType, idType> hash_map(map_ids, map_values, matrix_a.rows, trunk_num_c);
  // symbolic
  (spgemm_symbolic_kernel<T, I, 256, 32, 64>)<<<512, 256>>>(
      matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, bitset_b.row_ptr, bitset_b.ids_ptr, bitset_b.col_idx_ptr,
      bitset_b.use_count, row_trunk_ptr, hash_map.begins, hash_map.nexts, hash_map.ids, hash_map.values, map_use_count);

  // calc row_ptr
  (row_nnz_kernel<I, bitType, 32, 64>
   <<<512, 256>>>(matrix_a.rows, row_trunk_ptr, hash_map.values, map_use_count, row_ptr_c));

  // reduce than fan
  scan_in_place<I>(matrix_a.rows + 1, row_ptr_c);
  // set nnz of matrix c
  KOKKOS_HIP_CHECK(hipMemcpy(nnz_c, row_ptr_c + matrix_a.rows, sizeof(I), hipMemcpyDeviceToHost))
  // free
  KOKKOS_HIP_CHECK(hipFree(map_use_count))
  KOKKOS_HIP_CHECK(hipFree(map_values))
  KOKKOS_HIP_CHECK(hipFree(map_ids))
  KOKKOS_HIP_CHECK(hipFree(row_trunk_ptr))

#ifdef DEV_MODE
  I *host_row_ptr_c = new I[matrix_a.rows + 1];
  KOKKOS_HIP_CHECK(hipMemcpy(host_row_ptr_c, row_ptr_c, (matrix_a.rows + 1) * sizeof(I), hipMemcpyDeviceToHost))
  std::cout << "row result: " << std::endl;
  for (int i = 0; i < matrix_a.rows + 1; ++i) {
    std::cout << i << "\t:" << host_row_ptr_c[i] << std::endl;
  }
  delete[] host_row_ptr_c;
#endif
}

template <typename T, typename I>
void spgemm_numeric(const T alpha, const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b,
                    CsrMatrix<T, I> &matrix_c) {
  HashMap<T, I> hash_map(matrix_c.col_idx_ptr, matrix_c.values_ptr, matrix_c.rows, matrix_c.nnz);
  constexpr size_t N = 4;
  constexpr size_t LDS_MAP_CAP = 1 << 5;
  constexpr size_t LDS_HASH_MASK = LDS_MAP_CAP - 1;
  constexpr size_t GLOBAL_BLOCK_NUM = 1024;
  constexpr size_t BLOCK_THREAD_NUM = 64;
  constexpr size_t WF_SIZE = 64;
  (spgemm_numeric_kernel<T, I, N, LDS_MAP_CAP, LDS_HASH_MASK, BLOCK_THREAD_NUM,
                         WF_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
      alpha, matrix_a.rows, matrix_a.cols, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_a.values_ptr,
      matrix_b.row_ptr, matrix_b.col_idx_ptr, matrix_b.values_ptr, matrix_c.row_ptr, hash_map.begins, hash_map.nexts,
      hash_map.ids, hash_map.values);

#ifdef DEV_MODE
  I *host_row_ptr_c = new I[matrix_c.rows + 1];
  KOKKOS_HIP_CHECK(hipMemcpy(host_row_ptr_c, matrix_c.row_ptr, (matrix_c.rows + 1) * sizeof(I), hipMemcpyDeviceToHost))

  I *host_col_idx_ptr_c = new I[matrix_c.nnz];
  T *host_values_ptr_c = new T[matrix_c.nnz];
  KOKKOS_HIP_CHECK(
      hipMemcpy(host_col_idx_ptr_c, matrix_c.col_idx_ptr, (matrix_c.nnz) * sizeof(I), hipMemcpyDeviceToHost))
  KOKKOS_HIP_CHECK(hipMemcpy(host_values_ptr_c, matrix_c.values_ptr, (matrix_c.nnz) * sizeof(T), hipMemcpyDeviceToHost))
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

template <typename T, typename I> void rocsparse_sort_wrapper(CsrMatrix<T, I> &matrix) {
  using dtype = T;
  using csrIdxType = I;
  // Create rocsparse handle
  rocsparse_handle handle;
  ROCSPARSE_CHECK(rocsparse_create_handle(&handle));
  // Create rocsparse matrix descr
  rocsparse_mat_descr descr;
  ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr))

  // Create permutation vector perm as the identity map
  rocsparse_int *perm;
  KOKKOS_HIP_CHECK(hipMalloc((void **)&perm, sizeof(rocsparse_int) * matrix.nnz))
  ROCSPARSE_CHECK(rocsparse_create_identity_permutation(handle, matrix.nnz, perm))

  // Allocate temporary buffer
  size_t buffer_size;
  void *temp_buffer;
  ROCSPARSE_CHECK(rocsparse_csrsort_buffer_size(handle, matrix.rows, matrix.cols, matrix.nnz, matrix.row_ptr,
                                                matrix.col_idx_ptr, &buffer_size))
  KOKKOS_HIP_CHECK(hipMalloc(&temp_buffer, buffer_size))

  // Sort the CSR matrix
  ROCSPARSE_CHECK(rocsparse_csrsort(handle, matrix.rows, matrix.cols, matrix.nnz, descr, matrix.row_ptr,
                                    matrix.col_idx_ptr, perm, temp_buffer))

  // Free buffer
  KOKKOS_HIP_CHECK(hipFree(temp_buffer))
  ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr))
  ROCSPARSE_CHECK(rocsparse_destroy_handle(handle))
  // Gather sorted csr_val array
  dtype *csr_val_sorted;
  KOKKOS_HIP_CHECK(hipMalloc((void **)&csr_val_sorted, sizeof(dtype) * matrix.nnz));
  densetosparse_gthr_wrapper<dtype>(matrix.nnz, matrix.values_ptr, csr_val_sorted, perm, rocsparse_index_base_zero);

  // Clean up
  KOKKOS_HIP_CHECK(hipFree(perm))
  KOKKOS_HIP_CHECK(hipFree(matrix.values_ptr))

  // Reset values ptr
  matrix.values_ptr = csr_val_sorted;
}

template <typename T, typename I> void rocprim_sort_wrapper(CsrMatrix<T, I> &matrix) {
  size_t buffer_size;
  void *buffer = nullptr;
  unsigned int startbit = 0;
  unsigned int endbit = 64 - __builtin_clzll(matrix.cols);

  rocprim::segmented_radix_sort_pairs(buffer, buffer_size, matrix.col_idx_ptr, matrix.col_idx_ptr, matrix.values_ptr,
                                      matrix.values_ptr, matrix.nnz, matrix.rows, matrix.row_ptr, matrix.row_ptr + 1,
                                      startbit, endbit);

  KOKKOS_HIP_CHECK(hipMalloc((void **)&buffer, buffer_size))

  rocprim::segmented_radix_sort_pairs(buffer, buffer_size, matrix.col_idx_ptr, matrix.col_idx_ptr, matrix.values_ptr,
                                      matrix.values_ptr, matrix.nnz, matrix.rows, matrix.row_ptr, matrix.row_ptr + 1,
                                      startbit, endbit);
  KOKKOS_HIP_CHECK(hipFree(buffer))
}

template <typename T, typename I> void csr_matrix_sort(CsrMatrix<T, I> &matrix) {
  // rocprim_sort_wrapper<T, I>(matrix);
  rocsparse_sort_wrapper<T, I>(matrix);
}

#endif // SPGEMM_KOKKOS_SPGEMM_CORE_HPP
