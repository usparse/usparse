/**
 * Created by genshen on 2025/9/28.
 *
 */

#include "coomv_spmv_acc_wrappers.h"
#include "../../conversion/coo2csr.h"

#include "../csrmv_kernels/adaptive/adaptive.h"
#include "../csrmv_kernels/flat/csrmv_flat.h"
#include "../csrmv_kernels/light/spmv_hip_acc_imp.h"
#include "../csrmv_kernels/line-enhance/csrmv_line_enhance.h"
#include "../csrmv_kernels/vector-row/vector_row.h"
#include <cassert>

template <typename INDEX_TYPE, typename DATA_TYPE>
void usparse::spmv::internal::coomv_conv_analysis(const usparse::usparse_operation trans,
                                                  const usparse::coo_desc<INDEX_TYPE, DATA_TYPE> d_coo_desc,
                                                  coomv_conversion_mtx_info<INDEX_TYPE> *info,
                                                  const usparse::spmv::usparse_coomv_algorithms algo) {
  INDEX_TYPE *d_row_ptr;
  hipMalloc((void **)&d_row_ptr, sizeof(int) * (d_coo_desc.rows + 1));
  usparse::conversion::coo2csr(d_coo_desc.rows, d_coo_desc.nnz, d_coo_desc.row_index, d_row_ptr);
  info->csr_row_ptr = d_row_ptr;

  // flat method needs to generate break points.
  if (algo == usparse::spmv::usparse_coomv_algorithms::Flat) {
    const usparse::csr_desc<INDEX_TYPE, DATA_TYPE> dev_csr_mat = usparse::csr_desc<INDEX_TYPE, DATA_TYPE>(
        d_coo_desc.rows, d_coo_desc.cols, d_coo_desc.nnz, info->csr_row_ptr, d_coo_desc.col_index, d_coo_desc.values);
    info->flat_info = new csrmv_flat_mtx_info();
    auto d_csr_mat_descr = usparse::csr_mat_descr<DATA_TYPE, INDEX_TYPE, INDEX_TYPE, INDEX_TYPE>(dev_csr_mat);
    flat_analyse<DATA_TYPE, INDEX_TYPE, INDEX_TYPE, INDEX_TYPE, INDEX_TYPE>(d_csr_mat_descr, info->flat_info);
  }
}

template <typename INDEX_TYPE, typename DATA_TYPE>
void usparse::spmv::internal::coomv_conv_calculate(const usparse::usparse_operation trans, const DATA_TYPE alpha,
                                                   const DATA_TYPE beta,
                                                   const usparse::coo_desc<INDEX_TYPE, DATA_TYPE> d_coo_desc,
                                                   const DATA_TYPE *d_x, DATA_TYPE *d_y,
                                                   usparse::spmv::internal::coomv_conversion_mtx_info<INDEX_TYPE> *info,
                                                   const usparse::spmv::usparse_coomv_algorithms algo) {
  const usparse::csr_desc<INDEX_TYPE, DATA_TYPE> d_csr_desc = usparse::csr_desc<INDEX_TYPE, DATA_TYPE>(
      d_coo_desc.rows, d_coo_desc.cols, d_coo_desc.nnz, info->csr_row_ptr, d_coo_desc.col_index, d_coo_desc.values);

  if (algo == usparse::spmv::usparse_coomv_algorithms::Flat) {
    adaptive_flat_sparse_spmv<INDEX_TYPE, DATA_TYPE>(
        trans, alpha, beta, usparse::csr_mat_descr<DATA_TYPE, INDEX_TYPE, INDEX_TYPE, INDEX_TYPE>(d_csr_desc), d_x, d_y,
        info->flat_info);
  } else if (algo == usparse::spmv::usparse_coomv_algorithms::LineEnhance) {
    adaptive_enhance_sparse_spmv(trans, alpha, beta, d_csr_desc, d_x, d_y);
  } else if (algo == usparse::spmv::usparse_coomv_algorithms::Vector) {
    vec_row_sparse_spmv(trans, alpha, beta, d_csr_desc, d_x, d_y);
  } else {
    assert(false); // unsupported algorithm.
  }

  // TODO: add adaptive spmv.

  // light method is deprecated.
}

template <typename INDEX_TYPE>
void usparse::spmv::internal::coomv_conv_clear(usparse::spmv::internal::coomv_conversion_mtx_info<INDEX_TYPE> *info) {
  if (info == nullptr) {
    return;
  }

  hipFree(info->csr_row_ptr); // TODO: HIP_CHECK

  if (info->flat_info != nullptr) {
    // flat_clean(); TODO: release memory.
  }
  delete info->flat_info;
}

// instance the above template
template void usparse::spmv::internal::coomv_conv_analysis<usparse::usparse_int, usparse::usparse_fp64>(
    const usparse::usparse_operation trans,
    const usparse::coo_desc<usparse::usparse_int, usparse::usparse_fp64> d_coo_desc,
    coomv_conversion_mtx_info<usparse::usparse_int> *info, const usparse::spmv::usparse_coomv_algorithms algo);

template void usparse::spmv::internal::coomv_conv_analysis<usparse::usparse_int, usparse::usparse_fp32>(
    const usparse::usparse_operation trans,
    const usparse::coo_desc<usparse::usparse_int, usparse::usparse_fp32> d_coo_desc,
    coomv_conversion_mtx_info<usparse::usparse_int> *info, const usparse::spmv::usparse_coomv_algorithms algo);

template void usparse::spmv::internal::coomv_conv_calculate<usparse::usparse_int, usparse::usparse_fp32>(
    const usparse::usparse_operation trans, const usparse::usparse_fp32 alpha, const usparse::usparse_fp32 beta,
    const usparse::coo_desc<usparse::usparse_int, usparse::usparse_fp32> d_coo_desc, const usparse::usparse_fp32 *d_x,
    usparse::usparse_fp32 *d_y, usparse::spmv::internal::coomv_conversion_mtx_info<usparse::usparse_int> *info,
    const usparse::spmv::usparse_coomv_algorithms algo);

template void usparse::spmv::internal::coomv_conv_calculate<usparse::usparse_int, usparse::usparse_fp64>(
    const usparse::usparse_operation trans, const usparse::usparse_fp64 alpha, const usparse::usparse_fp64 beta,
    const usparse::coo_desc<usparse::usparse_int, usparse::usparse_fp64> d_coo_desc, const usparse::usparse_fp64 *d_x,
    usparse::usparse_fp64 *d_y, usparse::spmv::internal::coomv_conversion_mtx_info<usparse::usparse_int> *info,
    const usparse::spmv::usparse_coomv_algorithms algo);

template void usparse::spmv::internal::coomv_conv_clear<usparse::usparse_int>(
    usparse::spmv::internal::coomv_conversion_mtx_info<usparse::usparse_int> *info);
