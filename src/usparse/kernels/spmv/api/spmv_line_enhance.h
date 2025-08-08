//
// Created by yi on 2024/7/25.
//

#ifndef SPMV_LINE_ENHANCE_H
#define SPMV_LINE_ENHANCE_H

#include "building_config.h"
#include "types.h"

/**
 *
 * @param trans Matrix transpose. current only support operation_none.
 * @param alpha alpha and beta in y = alpha*A*x + beta*y
 * @param d_csr_desc csr description.
 * @param dx vector x.
 * @param dy vector y.
 */
template <typename T>
void spmv_line_enhance(int trans, const T alpha, const T beta, const csr_desc<int, T> d_csr_desc, const T *dx, T *dy);

#endif  // SPMV_LINE_ENHANCE_H