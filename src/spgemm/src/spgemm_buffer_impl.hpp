#ifndef USTB_SPARSE_SPGEMM_SPGEMM_BUFFER_IMPL_HPP
#define USTB_SPARSE_SPGEMM_SPGEMM_BUFFER_IMPL_HPP

#include "spgemm_adapt_policy.hpp"
#include "spgemm_numeric_impl.hpp"
#include "spgemm_symbolic_impl.hpp"

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
size_t spgemm_numeric_buffer_size(const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b) {
  return Policy::template buffer_size<T, I>(matrix_a, matrix_b);
}

template <typename T, typename I>
size_t spgemm_buffer_size(const PolicyFlag<T, I> &policyflag, const CsrMatrix<T, I> &matrix_a,
                          const CsrMatrix<T, I> &matrix_b) {
  size_t symbolic_buffer_size = 0;
  size_t numeric_buffer_size = 0;
  if (policyflag.use_compress) {
    symbolic_buffer_size = spgemm_symbolic_buffer_size<T, I, PreciseCompressSymbolicPolicy>(matrix_a, matrix_b);
  } else {
    symbolic_buffer_size = spgemm_symbolic_buffer_size<T, I, UpperBoundSymbolicPolicy>(matrix_a, matrix_b);
  }

  if (policyflag.use_bp) {
    if (policyflag.use_hash) {
      numeric_buffer_size = spgemm_numeric_buffer_size<T, I, HashMapNumericPolicy>(matrix_a, matrix_b);
    } else {
      numeric_buffer_size = spgemm_numeric_buffer_size<T, I, BreakPointNumericPolicy>(matrix_a, matrix_b);
    }
  } else {
    numeric_buffer_size = spgemm_numeric_buffer_size<T, I, EscNumericPolicy>(matrix_a, matrix_b);
  }

  return std::max<size_t>(symbolic_buffer_size, numeric_buffer_size);
}

#endif // USTB_SPARSE_SPGEMM_SPGEMM_BUFFER_IMPL_HPP
