#ifndef SPGEMM_KOKKOS_CREATE_BP_IMPL_HPP
#define SPGEMM_KOKKOS_CREATE_BP_IMPL_HPP

#include "create_bp_kernel.hpp"
#include "csr_matrix.h"

template <typename T, typename I, size_t CHUNK_SIZE> size_t bp_buffer_size(const CsrMatrix<T, I> &matrix) {
  const int BPS_PER_ROW = matrix.cols / CHUNK_SIZE + (matrix.cols % CHUNK_SIZE == 0 ? 0 : 1);
  const int BP_IDX_PER_ROW = BPS_PER_ROW + 1;
  const size_t buffer_size = sizeof(I) * matrix.rows * BP_IDX_PER_ROW;
  return buffer_size;
}

template <typename T, typename I, size_t CHUNK_SIZE>
void create_bp(const CsrMatrix<T, I> &matrix, I *bp, const size_t BP_IDX_PER_ROW) {
  create_bp_wf_per_row_kernel<I, CHUNK_SIZE>
      <<<512, 256>>>(matrix.rows, matrix.row_ptr, matrix.col_idx_ptr, bp, BP_IDX_PER_ROW);
}

#endif // SPGEMM_KOKKOS_CREATE_BP_IMPL_HPP