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
 * @param info the matrix info
 */
template <typename T>
void spmv_flat(int trans, const T alpha, const T beta, const csr_desc<int, T> d_csr_desc, const T *dx, T *dy,
               const csrmv_flat_mtx_info *info);

template <typename T> void flat_analyse(const csr_desc<int, T> d_csr_desc, int **break_points);

#endif  // SPMV_FLAT_H