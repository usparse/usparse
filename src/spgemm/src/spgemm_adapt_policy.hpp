#ifndef USTB_SPARSE_SPGEMM_SPGEMM_ADAPT_POLICY_HPP
#define USTB_SPARSE_SPGEMM_SPGEMM_ADAPT_POLICY_HPP

#include "compress_matrix.h"
#include "csr_matrix.h"
#include "spgemm_numeric_impl.hpp"
#include "spgemm_symbolic_impl.hpp"

template <typename T, typename I> struct PolicyFlag {
  double sparseness_a;
  double sparseness_b;
  double sparseness_c;
  bool SymbolicUpboundFlag = false;
  bool SymbolicCompressHashFlag = false;
  bool SymbolicCompressBitMapFlag = false;
  bool NumericBreakpointFlag = false;
  bool NumericHashFlag = false;
  bool NumericEscFlag = false;

  PolicyFlag(const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b) {
    sparseness_a = static_cast<double>(matrix_a.nnz) / static_cast<double>(matrix_a.rows * matrix_a.cols);
    sparseness_b = static_cast<double>(matrix_b.nnz) / static_cast<double>(matrix_b.rows * matrix_b.cols);
    if (sparseness_b < 0.005) {
      SymbolicUpboundFlag = true;
      NumericEscFlag = true;
    } else {
      if (matrix_b.cols <= 2048 * CompressMatrix<I>::BitsLen) {
        SymbolicCompressBitMapFlag = true;
      } else {
        SymbolicCompressHashFlag = true;
      }
    }
    // use_compress = sparseness_b > 0.005 ? true : false;
    // use_bp = sparseness_b > 0.005 ? true : false;
    // use_hash = matrix_a.rows < 10000 && sparseness_b < 0.015 ? true : false;
  }
};

class AdaptPolicyChoice {
public:
  template <typename T, typename I>
  static void symbolic_stage(const PolicyFlag<T, I> &policyflag, const CsrMatrix<T, I> &matrix_a,
                             const CsrMatrix<T, I> &matrix_b, I *row_ptr_c, size_t *nnz_c, I *row_nnz_max,
                             void *temp_buffer) {
    if (policyflag.SymbolicUpboundFlag) {
      spgemm_symbolic<T, I, UpperBoundSymbolicPolicy>(matrix_a, matrix_b, row_ptr_c, nnz_c, row_nnz_max, temp_buffer);
    } else if (policyflag.SymbolicCompressBitMapFlag) {
      spgemm_symbolic<T, I, CompressBitMapSymbolicPolicy>(matrix_a, matrix_b, row_ptr_c, nnz_c, row_nnz_max,
                                                          temp_buffer);
    } else {
      spgemm_symbolic<T, I, CompressHashSymbolicPolicy>(matrix_a, matrix_b, row_ptr_c, nnz_c, row_nnz_max, temp_buffer);
    }
  }

  template <typename T, typename I>
  static void numeric_stage(const PolicyFlag<T, I> &policyflag, const T alpha, const CsrMatrix<T, I> &matrix_a,
                            const CsrMatrix<T, I> &matrix_b, const I row_nnz_max, CsrMatrix<T, I> &matrix_c,
                            void *temp_buffer) {
    if (policyflag.NumericEscFlag) {
      spgemm_numeric<T, I, EscNumericPolicy>(alpha, matrix_a, matrix_b, row_nnz_max, matrix_c, temp_buffer);
    } else {
      spgemm_numeric<T, I, BreakPointNumericPolicy>(alpha, matrix_a, matrix_b, row_nnz_max, matrix_c, temp_buffer);
    }
  }
};

#endif // USTB_SPARSE_SPGEMM_SPGEMM_ADAPT_POLICY_HPP
