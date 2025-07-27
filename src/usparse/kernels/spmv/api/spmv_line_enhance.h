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
void spmv_line_enhance(int trans, const double alpha, const double beta,
                       const csr_desc<int, double> d_csr_desc, const double *dx,
                       double *dy);

#endif  // SPMV_LINE_ENHANCE_H