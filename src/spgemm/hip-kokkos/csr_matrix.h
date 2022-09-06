#ifndef SPGEMM_KOKKOS_CSR_MATRIX_H
#define SPGEMM_KOKKOS_CSR_MATRIX_H

template <typename T, typename I> struct CsrMatrix {
  const size_t rows;
  const size_t cols;
  const size_t nnz;
  I *row_ptr;
  I *col_idx_ptr;
  T *values_ptr;
  CsrMatrix(size_t m, size_t n, size_t nnz, I *r, I *c, T *v)
      : rows(m), cols(n), nnz(nnz), row_ptr(r), col_idx_ptr(c), values_ptr(v) {}
};

#endif // SPGEMM_KOKKOS_CSR_MATRIX_H