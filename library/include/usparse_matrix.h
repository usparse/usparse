//
// Created by genshen on 2025/05/08.
//
// Note: this file is copied from spmv-acc (ref: src/acc/api/types.h).
//

#ifndef USPARSE_HIP_USPARSE_MATRIX_H
#define USPARSE_HIP_USPARSE_MATRIX_H

namespace usparse {
  
  template <typename I, typename T> class coo_desc;
  template <typename I, typename A, typename J, typename K> class csr_desc;

  template <typename A, typename NNZ_TYPE, typename ROW_TYPE, typename COLUMN_TYPE>
  struct csr_mat_descr;

  /**
   * different from class csr_desc, all member is not const.
   * I,J are index types, A is value type in csr matrix.
   * CSR sparse matrix descriptor.
   */
  template <typename NNZ_TYPE, typename A, typename ROW_TYPE = NNZ_TYPE, typename COLUMN_TYPE = ROW_TYPE>
  class var_csr_desc {
  public:
    ROW_TYPE rows = 0;
    COLUMN_TYPE cols = 0;
    NNZ_TYPE nnz = 0;

    NNZ_TYPE *row_ptr = nullptr;
    COLUMN_TYPE *col_index = nullptr;
    A *values = nullptr;

    var_csr_desc() {}

    var_csr_desc(const ROW_TYPE r, const COLUMN_TYPE c, const NNZ_TYPE _nnz) : rows(r), cols(c), nnz(_nnz) {}

    csr_desc<NNZ_TYPE, A, ROW_TYPE, COLUMN_TYPE> as_const() {
      return csr_desc<NNZ_TYPE, A, ROW_TYPE, COLUMN_TYPE>(*this);
    }
  };

  /**
   * @deprecated API. use csr_mat_descr instead.
   *
   * CSR sparse matrix descriptor.
   * in class csr_desc, all members are const.
   * @tparam NNZ_TYPE index type for number of non-zeros.
   * @tparam ROW_TYPE and COLUMN_TYPE are index type of matrix row and column.
   * @tparam A is value type in csr matrix.
   */
  template <typename NNZ_TYPE, typename A, typename ROW_TYPE = NNZ_TYPE, typename COLUMN_TYPE = ROW_TYPE> class csr_desc {
  public:
    const ROW_TYPE rows = 0;
    const COLUMN_TYPE cols = 0;
    const NNZ_TYPE nnz = 0;

    const NNZ_TYPE *row_ptr = nullptr;
    const COLUMN_TYPE *col_index = nullptr;
    const A *values = nullptr;

    csr_desc(const ROW_TYPE m, const COLUMN_TYPE n, const NNZ_TYPE _nnz, const NNZ_TYPE *_row_ptr,
             const COLUMN_TYPE *_col_index, const A *_values)
        : rows(m), cols(n), nnz(_nnz), row_ptr(_row_ptr), col_index(_col_index), values(_values) {}

    csr_desc(const var_csr_desc<NNZ_TYPE, A, ROW_TYPE, COLUMN_TYPE> _var_csr_desc)
        : rows(_var_csr_desc.rows), cols(_var_csr_desc.cols), nnz(_var_csr_desc.nnz), row_ptr(_var_csr_desc.row_ptr),
          col_index(_var_csr_desc.col_index), values(_var_csr_desc.values) {}
  };

  /**
   * CSR sparse matrix descriptor.
   * @tparam A is value type in csr matrix.
   * @tparam NNZ_TYPE index type for number of non-zeros.
   * @tparam ROW_TYPE and COLUMN_TYPE are index type of matrix row and column.
   */
  template <typename A, typename NNZ_TYPE, typename ROW_TYPE = NNZ_TYPE, typename COLUMN_TYPE = ROW_TYPE>
  struct csr_mat_descr : public csr_desc<NNZ_TYPE, A, ROW_TYPE, COLUMN_TYPE> {
    using csr_desc<NNZ_TYPE, A, ROW_TYPE, COLUMN_TYPE>::csr_desc;
    // convert from csr_desc to csr_mat_descr:
    csr_mat_descr(const csr_desc<NNZ_TYPE, A, ROW_TYPE, COLUMN_TYPE> &desc)
        : csr_mat_descr<A, NNZ_TYPE, ROW_TYPE, COLUMN_TYPE>(desc.rows, desc.cols, desc.nnz, desc.row_ptr,
                                                            desc.col_index, desc.values) {}
  };

  /**
   * different from class coo_desc, all member is not const.
   * I is index type, T is value type in csr matrix.
   * COO sparse matrix descriptor.
   */
  template <typename I, typename T> class var_coo_desc {
  public:
    I rows = 0;
    I cols = 0;
    I nnz = 0;

    I *row_index = nullptr;
    I *col_index = nullptr;
    T *values = nullptr;

    var_coo_desc() {}

    var_coo_desc(const I r, const I c, const T _nnz) : rows(r), cols(c), nnz(_nnz) {}

    coo_desc<I, T> as_const() { return coo_desc<I, T>(*this); }
  };

  /**
   * COO sparse matrix descriptor.
   * in class csr_desc, all members are const.
   * I is index type, T is value type in csr matrix.
   */
  template <typename I, typename T> class coo_desc {
  public:
    const I rows = 0;
    const I cols = 0;
    const I nnz = 0;

    const I *row_index = nullptr;
    const I *col_index = nullptr;
    const T *values = nullptr;

    coo_desc(const I m, const I n, const I _nnz, const I *_row_index, const I *_col_index, const T *_values)
        : rows(m), cols(n), nnz(_nnz), row_index(_row_index), col_index(_col_index), values(_values) {}

    coo_desc(const var_csr_desc<I, T> _var_coo_desc)
        : rows(_var_coo_desc.rows), cols(_var_coo_desc.cols), nnz(_var_coo_desc.nnz),
          row_index(_var_coo_desc.row_index), col_index(_var_coo_desc.col_index), values(_var_coo_desc.values) {}
  };

} // namespace usparse

// for backward compatibility of csrmv.
// todo: remove it.
template <typename I, typename T> using var_csr_desc = usparse::var_csr_desc<I, T>;

template <typename I, typename A, typename J = I> using csr_desc = usparse::csr_desc<I, A, J>;

template <typename A, typename NNZ_TYPE, typename ROW_TYPE = NNZ_TYPE, typename COLUMN_TYPE = ROW_TYPE>
using csr_mat_descr = usparse::csr_mat_descr<A, NNZ_TYPE, ROW_TYPE, COLUMN_TYPE>;

#endif // USPARSE_HIP_USPARSE_MATRIX_H
