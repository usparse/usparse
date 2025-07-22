//
// Created by yi on 2024/7/25.
//

#ifndef SPMV_FLAT_H
#define SPMV_FLAT_H

#include "building_config.h"
#include "types.h"

/**
 *
 * @param trans Matrix transpose. current only support operation_none.
 * @param alpha alpha and beta in y = alpha*A*x + beta*y
 * @param h_rowptr row offset on host size.
 * @param d_csr_desc csr description.
 * @param dx vector x.
 * @param dy vector y.
 */
void spmv_flat(int trans, const double alpha, const double beta,
               const csr_desc<int, double> h_csr_desc,
               const csr_desc<int, double> d_csr_desc, const double *dx,
               double *dy);

#endif  // SPMV_FLAT_H