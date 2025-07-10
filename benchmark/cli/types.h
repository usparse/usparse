//
// Created by genshen on 2021/6/29.
//

#ifndef SPMV_ACC_TYPES_H
#define SPMV_ACC_TYPES_H

template <typename I, typename T> class csr_desc;

template <typename I, typename T> class var_csr_desc {
public:
  I rows = 0;
  I cols = 0;
  I nnz = 0;

  I *row_ptr = nullptr;
  I *col_index = nullptr;
  T *values = nullptr;

  csr_desc<I, T> as_const() { return csr_desc<I, T>(*this); }
};

template <typename I, typename T> class csr_desc {
public:
  const I rows = 0;
  const I cols = 0;
  const I nnz = 0;

  const I *row_ptr = nullptr;
  const I *col_index = nullptr;
  const T *values = nullptr;

  csr_desc(const I m, const I n, const I _nnz, const I *_row_ptr, const I *_col_index, const T *_values)
      : rows(m), cols(n), nnz(_nnz), row_ptr(_row_ptr), col_index(_col_index), values(_values) {}

  csr_desc(const var_csr_desc<I, T> _var_csr_desc)
      : rows(_var_csr_desc.rows), cols(_var_csr_desc.cols), nnz(_var_csr_desc.nnz), row_ptr(_var_csr_desc.row_ptr),
        col_index(_var_csr_desc.col_index), values(_var_csr_desc.values) {}
};

// CSR sparse matrix type
template <typename I, typename T> class csr_mtx : public var_csr_desc<I, T> {
public:
  void alloc(I _rows, I _cols, I _nnz) {
    this->nnz = _nnz;
    this->rows = _rows;
    this->cols = _cols;

    this->row_ptr = new I[_rows + 1];
    this->col_index = new I[_nnz];
    this->values = new T[_nnz];
  }
};

#endif // SPMV_ACC_TYPES_H
