#ifndef SPGEMM_KOKKOS_SPGEMM_SYMBOLIC_IMPL_HPP
#define SPGEMM_KOKKOS_SPGEMM_SYMBOLIC_IMPL_HPP

#include "../common/check/hip_check.hpp"
#include "../common/max/max_impl.hpp"
#include "../common/scan/scan_impl.hpp"
#include "align.h"
#include "compress_matrix.h"
#include "csr_matrix.h"
#include "spgemm_compress_impl.hpp"
#include "spgemm_symbolic_kernel.hpp"
#include <cstddef>
#include <hip/hip_runtime.h>

class BreakPointSymbolicPolicy {
public:
  static constexpr int CHUNK_SIZE = 2048;
  static constexpr int BLOCK_THREAD_NUM = 512;
  static constexpr int VEC_SIZE = 16;

  template <typename T, typename I>
  static size_t buffer_size(const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b) {
    size_t alloc_size = alloc_size = bp_buffer_size<T, I, CHUNK_SIZE>(matrix_b);
    return ALIGN_256_BYTES(alloc_size);
  }

  template <typename T, typename I>
  static void symbolic_row_nnz(const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b, I *row_nnz_ptr_c,
                               void *max_temp_buffer, void *temp_buffer) {
    const int GLOBAL_BLOCK_NUM = matrix_a.rows;

    const int BPS_PER_ROW = matrix_b.cols / CHUNK_SIZE + (matrix_b.cols % CHUNK_SIZE == 0 ? 0 : 1);
    const int BP_IDX_PER_ROW = BPS_PER_ROW + 1;

    char *ptr = reinterpret_cast<char *>(temp_buffer);
    size_t alloc_size = 0;
    size_t align_size = 0;

    // alloc buffer for break point
    I *bp = reinterpret_cast<I *>(ptr);
    alloc_size = bp_buffer_size<T, I, CHUNK_SIZE>(matrix_b);
    align_size = ALIGN_256_BYTES(alloc_size);
    ptr += align_size;

    create_bp<T, I, CHUNK_SIZE>(matrix_b, bp, BP_IDX_PER_ROW);

    (spgemm_symbolic_block_per_row_bp_kernel<T, I, CHUNK_SIZE, BLOCK_THREAD_NUM,
                                             VEC_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
        matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_b.row_ptr, matrix_b.col_idx_ptr, row_nnz_ptr_c,
        bp, BPS_PER_ROW, BP_IDX_PER_ROW);
  }
};

class PreciseNormalSymbolicPolicy {
public:
  template <typename T, typename I>
  static size_t buffer_size(const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b) {
    size_t buffer_size = 0;
    size_t alloc_size = 0;

    alloc_size = (matrix_a.rows + 1) * sizeof(I);
    buffer_size += ALIGN_256_BYTES(alloc_size);

    alloc_size = sizeof(I);
    buffer_size += ALIGN_256_BYTES(alloc_size);
    return buffer_size;
  }

  template <typename T, typename I, typename AlternativePolicy = BreakPointSymbolicPolicy>
  static void symbolic_row_nnz(const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b, I *row_nnz_ptr_c,
                               void *max_temp_buffer, void *temp_buffer) {
    char *ptr = reinterpret_cast<char *>(temp_buffer);
    size_t alloc_size = 0;
    size_t align_size = 0;

    // calc size of chunk
    I *chunk_num_ptr = reinterpret_cast<I *>(ptr);
    alloc_size = (matrix_a.rows + 1) * sizeof(I);
    align_size = ALIGN_256_BYTES(alloc_size);
    ptr += align_size;
    (chunk_nnz_kernel<I, 8>)<<<512, 256>>>(matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_b.row_ptr,
                                           chunk_num_ptr);

    // max of chunk num
    I hchunk_nnz_max = static_cast<I>(0);
    {
      I *chunk_num_max_ptr = reinterpret_cast<I *>(ptr);
      alloc_size = sizeof(I);
      align_size = ALIGN_256_BYTES(alloc_size);
      ptr += align_size;
      common::max<I>(matrix_a.rows, chunk_num_ptr + 1, chunk_num_max_ptr, max_temp_buffer);
      // HIP_CHECK(hipMemcpy(&hchunk_nnz_max, chunk_num_max_ptr, sizeof(I), hipMemcpyDeviceToHost))
      hipDeviceSynchronize();
      hchunk_nnz_max = *chunk_num_max_ptr;
    }

    hchunk_nnz_max = std::min<I>(matrix_b.cols, hchunk_nnz_max);
    if (hchunk_nnz_max <= 128) {
      constexpr size_t N = 4;
      constexpr size_t VEC_SIZE = 128 / N;
      constexpr size_t BT_SIZE = 8;
      constexpr size_t BLOCK_THREAD_NUM = 256;
      constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
      const size_t GLOBAL_BLOCK_NUM = matrix_a.rows / BLOCK_VEC_NUM + (matrix_a.rows % BLOCK_VEC_NUM == 0 ? 0 : 1);

      (spgemm_symbolic_open_address_hashmap_vector_per_row_one_pass_kernel<
          T, I, N, BLOCK_THREAD_NUM, VEC_SIZE, BT_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_b.row_ptr, matrix_b.col_idx_ptr, row_nnz_ptr_c);

    } else if (hchunk_nnz_max <= 512) {
      constexpr size_t N = 2;
      constexpr size_t BLOCK_THREAD_NUM = 512 / N;
      const size_t GLOBAL_BLOCK_NUM = matrix_a.rows;

      (spgemm_symbolic_open_address_hashmap_block_per_row_one_pass_kernel<
          T, I, N, BLOCK_THREAD_NUM>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_b.row_ptr, matrix_b.col_idx_ptr, row_nnz_ptr_c);

    } else if (hchunk_nnz_max <= 1024) {
      constexpr size_t N = 4;
      constexpr size_t BLOCK_THREAD_NUM = 1024 / N;
      const size_t GLOBAL_BLOCK_NUM = matrix_a.rows;

      (spgemm_symbolic_open_address_hashmap_block_per_row_one_pass_kernel<
          T, I, N, BLOCK_THREAD_NUM>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_b.row_ptr, matrix_b.col_idx_ptr, row_nnz_ptr_c);

    } else if (hchunk_nnz_max <= 2048) {
      constexpr size_t N = 8;
      constexpr size_t BLOCK_THREAD_NUM = 2048 / N;
      const size_t GLOBAL_BLOCK_NUM = matrix_a.rows;

      (spgemm_symbolic_open_address_hashmap_block_per_row_one_pass_kernel<
          T, I, N, BLOCK_THREAD_NUM>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_b.row_ptr, matrix_b.col_idx_ptr, row_nnz_ptr_c);

    } else {
      // if row nnz max larger than 2048, turn to other method.
      const size_t precise_normal_buffer_size = buffer_size<T, I>(matrix_a, matrix_b);
      const size_t alternative_policy_buffer_size = AlternativePolicy::template buffer_size<T, I>(matrix_a, matrix_b);
      if (precise_normal_buffer_size >= alternative_policy_buffer_size) {
        AlternativePolicy::template symbolic_row_nnz<T, I>(matrix_a, matrix_b, row_nnz_ptr_c, max_temp_buffer,
                                                           temp_buffer);
      } else {
        void *alternative_buffer = nullptr;
        HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&alternative_buffer), alternative_policy_buffer_size))
        AlternativePolicy::template symbolic_row_nnz<T, I>(matrix_a, matrix_b, row_nnz_ptr_c, max_temp_buffer,
                                                           alternative_buffer);
        HIP_CHECK(hipFree(alternative_buffer))
      }
    }
  }
};

class PreciseCompressSymbolicPolicy {
public:
  template <typename T, typename I>
  static size_t buffer_size(const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b) {
    size_t buffer_size = 0;
    size_t alloc_size = 0;

    alloc_size = spgemm_compress_buffer_size(matrix_b);
    buffer_size += ALIGN_256_BYTES(alloc_size);

    alloc_size = (matrix_a.rows + 1) * sizeof(I);
    buffer_size += ALIGN_256_BYTES(alloc_size);

    alloc_size = sizeof(I);
    buffer_size += ALIGN_256_BYTES(alloc_size);

    return buffer_size;
  }

  template <typename T, typename I, typename AlternativePolicy = BreakPointSymbolicPolicy>
  static void symbolic_row_nnz(const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b, I *row_nnz_ptr_c,
                               void *max_temp_buffer, void *temp_buffer) {
    char *ptr = reinterpret_cast<char *>(temp_buffer);
    size_t alloc_size = 0;
    size_t align_size = 0;
    // compress matrix B
    CompressMatrix<I> compress_matrix_b = spgemm_compress<T, I>(matrix_b, reinterpret_cast<void *>(ptr));

    alloc_size = spgemm_compress_buffer_size(matrix_b);
    align_size = ALIGN_256_BYTES(alloc_size);
    ptr += align_size;

    // max num in bits
    I max_bits_num =
        matrix_b.cols / CompressMatrix<I>::BitsLen + (matrix_b.cols % CompressMatrix<I>::BitsLen == 0 ? 0 : 1);

    if (max_bits_num <= 16) {
      constexpr size_t N = 2;
      constexpr size_t VEC_SIZE = 16 / N;
      constexpr size_t BLOCK_THREAD_NUM = 256;
      constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
      const size_t GLOBAL_BLOCK_NUM = matrix_a.rows / BLOCK_VEC_NUM + (matrix_a.rows % BLOCK_VEC_NUM == 0 ? 0 : 1);

      (spgemm_symbolic_vector_per_row_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM,
                                                      VEC_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr, compress_matrix_b.ids_ptr,
          compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_nnz_ptr_c);
    } else if (max_bits_num <= 32) {
      constexpr size_t N = 2;
      constexpr size_t VEC_SIZE = 32 / N;
      constexpr size_t BLOCK_THREAD_NUM = 256;
      constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
      const size_t GLOBAL_BLOCK_NUM = matrix_a.rows / BLOCK_VEC_NUM + (matrix_a.rows % BLOCK_VEC_NUM == 0 ? 0 : 1);

      (spgemm_symbolic_vector_per_row_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM,
                                                      VEC_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr, compress_matrix_b.ids_ptr,
          compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_nnz_ptr_c);
    } else if (max_bits_num <= 64) {
      constexpr size_t N = 2;
      constexpr size_t VEC_SIZE = 64 / N;
      constexpr size_t BLOCK_THREAD_NUM = 256;
      constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
      const size_t GLOBAL_BLOCK_NUM = matrix_a.rows / BLOCK_VEC_NUM + (matrix_a.rows % BLOCK_VEC_NUM == 0 ? 0 : 1);

      (spgemm_symbolic_vector_per_row_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM,
                                                      VEC_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr, compress_matrix_b.ids_ptr,
          compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_nnz_ptr_c);
    } else if (max_bits_num <= 128) {
      constexpr size_t N = 2;
      constexpr size_t VEC_SIZE = 128 / N;
      constexpr size_t BLOCK_THREAD_NUM = 256;
      constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
      const size_t GLOBAL_BLOCK_NUM = matrix_a.rows / BLOCK_VEC_NUM + (matrix_a.rows % BLOCK_VEC_NUM == 0 ? 0 : 1);

      (spgemm_symbolic_vector_per_row_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM,
                                                      VEC_SIZE>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr, compress_matrix_b.ids_ptr,
          compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_nnz_ptr_c);
    } else if (max_bits_num <= 512) {
      constexpr size_t N = 2;
      constexpr size_t BLOCK_THREAD_NUM = 512 / N;
      const size_t GLOBAL_BLOCK_NUM = matrix_a.rows;

      (spgemm_symbolic_block_per_row_one_pass_kernel<T, I, N,
                                                     BLOCK_THREAD_NUM>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr, compress_matrix_b.ids_ptr,
          compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_nnz_ptr_c);
    } else if (max_bits_num <= 1024) {
      constexpr size_t N = 2;
      constexpr size_t BLOCK_THREAD_NUM = 1024 / N;
      const size_t GLOBAL_BLOCK_NUM = matrix_a.rows;

      (spgemm_symbolic_block_per_row_one_pass_kernel<T, I, N,
                                                     BLOCK_THREAD_NUM>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr, compress_matrix_b.ids_ptr,
          compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_nnz_ptr_c);
    } else if (max_bits_num <= 2048) {
      constexpr size_t N = 2;
      constexpr size_t BLOCK_THREAD_NUM = 2048 / N;
      const size_t GLOBAL_BLOCK_NUM = matrix_a.rows;

      (spgemm_symbolic_block_per_row_one_pass_kernel<T, I, N,
                                                     BLOCK_THREAD_NUM>)<<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
          matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr, compress_matrix_b.ids_ptr,
          compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_nnz_ptr_c);
    } else {
      // calc size of chunk
      I *chunk_num_ptr = reinterpret_cast<I *>(ptr);
      alloc_size = (matrix_a.rows + 1) * sizeof(I);
      align_size = ALIGN_256_BYTES(alloc_size);
      ptr += align_size;

      (chunk_nnz_compress_kernel<I, 4>)<<<512, 256>>>(matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr,
                                                      compress_matrix_b.use_count, chunk_num_ptr);

      // max of chunk num
      I hchunk_nnz_max = static_cast<I>(0);
      {
        I *chunk_num_max_ptr = reinterpret_cast<I *>(ptr);
        alloc_size = sizeof(I);
        align_size = ALIGN_256_BYTES(alloc_size);
        ptr += align_size;
        common::max<I>(matrix_a.rows, chunk_num_ptr + 1, chunk_num_max_ptr, max_temp_buffer);
        // HIP_CHECK(hipMemcpy(&hchunk_nnz_max, chunk_num_max_ptr, sizeof(I), hipMemcpyDeviceToHost))
        hipDeviceSynchronize();
        hchunk_nnz_max = *chunk_num_max_ptr;
      }

      hchunk_nnz_max = std::min<I>(matrix_b.cols, hchunk_nnz_max);

      if (hchunk_nnz_max <= 16) {
        constexpr size_t N = 2;
        constexpr size_t VEC_SIZE = 16 / N;
        constexpr size_t BLOCK_THREAD_NUM = 256;
        constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
        const size_t GLOBAL_BLOCK_NUM = matrix_a.rows / BLOCK_VEC_NUM + (matrix_a.rows % BLOCK_VEC_NUM == 0 ? 0 : 1);

        spgemm_symbolic_open_address_hashmap_vector_per_row_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM, VEC_SIZE>
            <<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
                matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr,
                compress_matrix_b.ids_ptr, compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_nnz_ptr_c);

      } else if (hchunk_nnz_max <= 32) {
        constexpr size_t N = 2;
        constexpr size_t VEC_SIZE = 32 / N;
        constexpr size_t BLOCK_THREAD_NUM = 256;
        constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
        const size_t GLOBAL_BLOCK_NUM = matrix_a.rows / BLOCK_VEC_NUM + (matrix_a.rows % BLOCK_VEC_NUM == 0 ? 0 : 1);

        spgemm_symbolic_open_address_hashmap_vector_per_row_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM, VEC_SIZE>
            <<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
                matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr,
                compress_matrix_b.ids_ptr, compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_nnz_ptr_c);

      } else if (hchunk_nnz_max <= 64) {
        constexpr size_t N = 2;
        constexpr size_t VEC_SIZE = 64 / N;
        constexpr size_t BLOCK_THREAD_NUM = 256;
        constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
        const size_t GLOBAL_BLOCK_NUM = matrix_a.rows / BLOCK_VEC_NUM + (matrix_a.rows % BLOCK_VEC_NUM == 0 ? 0 : 1);

        spgemm_symbolic_open_address_hashmap_vector_per_row_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM, VEC_SIZE>
            <<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
                matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr,
                compress_matrix_b.ids_ptr, compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_nnz_ptr_c);

      } else if (hchunk_nnz_max <= 128) {
        constexpr size_t N = 2;
        constexpr size_t VEC_SIZE = 128 / N;
        constexpr size_t BLOCK_THREAD_NUM = 256;
        constexpr size_t BLOCK_VEC_NUM = BLOCK_THREAD_NUM / VEC_SIZE;
        const size_t GLOBAL_BLOCK_NUM = matrix_a.rows / BLOCK_VEC_NUM + (matrix_a.rows % BLOCK_VEC_NUM == 0 ? 0 : 1);

        spgemm_symbolic_open_address_hashmap_vector_per_row_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM, VEC_SIZE>
            <<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
                matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr,
                compress_matrix_b.ids_ptr, compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_nnz_ptr_c);

      } else if (hchunk_nnz_max <= 512) {
        constexpr size_t N = 2;
        constexpr size_t BLOCK_THREAD_NUM = 512 / N;
        const size_t GLOBAL_BLOCK_NUM = matrix_a.rows;
        spgemm_symbolic_open_address_hashmap_block_per_row_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM>
            <<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
                matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr,
                compress_matrix_b.ids_ptr, compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_nnz_ptr_c);

      } else if (hchunk_nnz_max <= 1024) {
        constexpr size_t N = 2;
        constexpr size_t BLOCK_THREAD_NUM = 1024 / N;
        const size_t GLOBAL_BLOCK_NUM = matrix_a.rows;
        spgemm_symbolic_open_address_hashmap_block_per_row_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM>
            <<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
                matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr,
                compress_matrix_b.ids_ptr, compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_nnz_ptr_c);

      } else if (hchunk_nnz_max <= 2048) {
        constexpr size_t N = 2;
        constexpr size_t BLOCK_THREAD_NUM = 2048 / N;
        const size_t GLOBAL_BLOCK_NUM = matrix_a.rows;
        spgemm_symbolic_open_address_hashmap_block_per_row_one_pass_kernel<T, I, N, BLOCK_THREAD_NUM>
            <<<GLOBAL_BLOCK_NUM, BLOCK_THREAD_NUM>>>(
                matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, compress_matrix_b.row_ptr,
                compress_matrix_b.ids_ptr, compress_matrix_b.col_idx_ptr, compress_matrix_b.use_count, row_nnz_ptr_c);

      } else {
        // if row nnz max larger than 2048, turn to other method.
        const size_t precise_compress_buffer_size = buffer_size<T, I>(matrix_a, matrix_b);
        const size_t alternative_policy_buffer_size = AlternativePolicy::template buffer_size<T, I>(matrix_a, matrix_b);
        if (precise_compress_buffer_size >= alternative_policy_buffer_size) {
          AlternativePolicy::template symbolic_row_nnz<T, I>(matrix_a, matrix_b, row_nnz_ptr_c, max_temp_buffer,
                                                             temp_buffer);
        } else {
          void *alternative_buffer = nullptr;
          HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&alternative_buffer), alternative_policy_buffer_size))
          AlternativePolicy::template symbolic_row_nnz<T, I>(matrix_a, matrix_b, row_nnz_ptr_c, max_temp_buffer,
                                                             alternative_buffer);
          HIP_CHECK(hipFree(alternative_buffer))
        }
      }
    }
  }
};

class UpperBoundSymbolicPolicy {
public:
  template <typename T, typename I>
  static size_t buffer_size(const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b) {
    return 0;
  }
  template <typename T, typename I>
  static void symbolic_row_nnz(const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b, I *row_nnz_ptr_c,
                               void *max_temp_buffer, void *temp_buffer) {
    (chunk_nnz_kernel<I, 8>)<<<512, 256>>>(matrix_a.rows, matrix_a.row_ptr, matrix_a.col_idx_ptr, matrix_b.row_ptr,
                                           row_nnz_ptr_c);
  }
};

template <typename T, typename I, typename Policy>
size_t spgemm_symbolic_buffer_size(const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b) {
  size_t buffer_size = 0;
  size_t alloc_size = 0;

  // buffer for scan op
  alloc_size = common::scan_buffer_size<I>(matrix_a.rows + 1);
  buffer_size += ALIGN_256_BYTES(alloc_size);
  // buffer for max op
  alloc_size = common::max_buffer_size<I>(matrix_a.rows + 1);
  buffer_size += ALIGN_256_BYTES(alloc_size);
  // buffer for max nnz item
  alloc_size = sizeof(I);
  buffer_size += ALIGN_256_BYTES(alloc_size);
  // buffer for row nnz policy
  alloc_size = Policy::template buffer_size<T, I>(matrix_a, matrix_b);
  buffer_size += ALIGN_256_BYTES(alloc_size);

  return buffer_size;
}

template <typename T, typename I, typename Policy>
void spgemm_symbolic(const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b, I *row_ptr_c, size_t *hnnz_c,
                     I *hrow_nnz_max, void *temp_buffer) {
  char *ptr = reinterpret_cast<char *>(temp_buffer);
  size_t alloc_size = 0;
  size_t align_size = 0;

  // alloc buffer for scan op
  void *scan_temp_buffer = reinterpret_cast<void *>(ptr);
  alloc_size = common::scan_buffer_size<I>(matrix_a.rows + 1);
  align_size = ALIGN_256_BYTES(alloc_size);
  ptr += align_size;
  // alloc buffer for max op
  void *max_temp_buffer = reinterpret_cast<void *>(ptr);
  alloc_size = common::max_buffer_size<I>(matrix_a.rows + 1);
  align_size = ALIGN_256_BYTES(alloc_size);
  ptr += align_size;
  // alloc buffer for max nnz item
  I *row_nnz_max_ptr = reinterpret_cast<I *>(ptr);
  alloc_size = sizeof(I);
  align_size = ALIGN_256_BYTES(alloc_size);
  ptr += align_size;

  // reuse row_ptr_c
  I *row_nnz_ptr_c = reinterpret_cast<I *>(row_ptr_c);
  Policy::template symbolic_row_nnz<T, I>(matrix_a, matrix_b, row_nnz_ptr_c, max_temp_buffer,
                                          reinterpret_cast<void *>(ptr));

  // find max in row_nnz
  common::max<I>(matrix_a.rows + 1, row_nnz_ptr_c, row_nnz_max_ptr, max_temp_buffer);
  // HIP_CHECK(hipMemcpy(hrow_nnz_max, row_nnz_max_ptr, sizeof(I), hipMemcpyDeviceToHost))
  // scan row_nnz -> row_ptr_c
  common::scan<I>(matrix_a.rows + 1, row_nnz_ptr_c, row_ptr_c, scan_temp_buffer);
  // HIP_CHECK(hipMemcpy(hnnz_c, row_ptr_c + matrix_a.rows, sizeof(I), hipMemcpyDeviceToHost))
  hipDeviceSynchronize();
  *hrow_nnz_max = *row_nnz_max_ptr;
  *hnnz_c = row_ptr_c[matrix_a.rows];
}

#endif // SPGEMM_KOKKOS_SPGEMM_SYMBOLIC_IMPL_HPP
