/**
 * Created by genshen on 2025/9/23.
 *
 */

#ifndef USPARSE_HIP_LEVEL2_COOMV_H
#define USPARSE_HIP_LEVEL2_COOMV_H

#include "../spmv_api_config.hpp"
#include "include/usparse.h"

template <typename INDEX_TYPE, typename DATA_TYPE, int BEAT_Y_POLICY = BEAT_Y_POLICY_AUTO>
void adaptive_flat_coo_sparse_spmv(int trans, const DATA_TYPE alpha, const DATA_TYPE beta,
                                   const usparse::coo_desc<INDEX_TYPE, DATA_TYPE> d_coo_desc, const DATA_TYPE *x,
                                   DATA_TYPE *y);

#endif // USPARSE_HIP_LEVEL2_COOMV_H
