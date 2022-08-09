#ifndef SPGEMM_KOKKOS_CSR_MATRIX_H
#define SPGEMM_KOKKOS_CSR_MATRIX_H

template <typename T, typename I> struct CsrMatrix {
  using dtype = T;
  using csrIdxType = I;
  const size_t rows;
  const size_t cols;
  const size_t nnz;
  csrIdxType *row_ptr;
  csrIdxType *col_idx_ptr;
  dtype *values_ptr;
  CsrMatrix(size_t m, size_t n, size_t nnz, csrIdxType *r, csrIdxType *c, dtype *v)
      : rows(m), cols(n), nnz(nnz), row_ptr(r), col_idx_ptr(c), values_ptr(v) {}
};

#endif // SPGEMM_KOKKOS_CSR_MATRIX_H