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
    } else {
      if (matrix_b.cols <= 2048 * CompressMatrix<I>::BitsLen) {
        SymbolicCompressBitMapFlag = true;
      } else {
        SymbolicCompressHashFlag = true;
      }
    }
  }

  void SetNumericPolicyFlag(const I row_nnz_max, const CsrMatrix<T, I> &matrix_a, const CsrMatrix<T, I> &matrix_b,
                            const CsrMatrix<T, I> &matrix_c, void *&temp_buffer, const size_t temp_buffer_size) {
    sparseness_c = static_cast<double>(matrix_c.nnz) / static_cast<double>(matrix_c.rows * matrix_c.cols);
    if (row_nnz_max <= 2048) {
      if (sparseness_b < 0.005) {
        NumericEscFlag = true;
      } else {
        if (sparseness_c <= 0.05) {
          NumericHashFlag = true;
        } else {
          NumericBreakpointFlag = true;
          if (temp_buffer_size < BreakPointNumericPolicy::template buffer_size<T, I>(matrix_a, matrix_b)) {
            HIP_CHECK(hipFree(temp_buffer));
            HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&temp_buffer),
                                BreakPointNumericPolicy::template buffer_size<T, I>(matrix_a, matrix_b)))
          }
        }
      }
    } else {
      NumericBreakpointFlag = true;
      if (temp_buffer_size < BreakPointNumericPolicy::template buffer_size<T, I>(matrix_a, matrix_b)) {
        HIP_CHECK(hipFree(temp_buffer));
        HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&temp_buffer),
                            BreakPointNumericPolicy::template buffer_size<T, I>(matrix_a, matrix_b)))
      }
    }
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
  static void numeric_stage(PolicyFlag<T, I> &policyflag, const T alpha, const CsrMatrix<T, I> &matrix_a,
                            const CsrMatrix<T, I> &matrix_b, const I row_nnz_max, CsrMatrix<T, I> &matrix_c,
                            void *&temp_buffer, const size_t temp_buffer_size) {
    policyflag.SetNumericPolicyFlag(row_nnz_max, matrix_a, matrix_b, matrix_c, temp_buffer, temp_buffer_size);
    if (policyflag.NumericEscFlag) {
      spgemm_numeric<T, I, EscNumericPolicy>(alpha, matrix_a, matrix_b, matrix_c, temp_buffer);
    } else if (policyflag.NumericHashFlag) {
      spgemm_numeric<T, I, HashMapNumericPolicy>(alpha, matrix_a, matrix_b, matrix_c, temp_buffer);
    } else {
      spgemm_numeric<T, I, BreakPointNumericPolicy>(alpha, matrix_a, matrix_b, matrix_c, temp_buffer);
    }
  }
};

#endif // USTB_SPARSE_SPGEMM_SPGEMM_ADAPT_POLICY_HPP