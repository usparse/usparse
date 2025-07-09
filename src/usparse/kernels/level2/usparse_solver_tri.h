//
// Created by genshen to 2025-07-09.
//

#ifndef USPARSE_KERNELS_LEVEL2_USPARSE_SOLVER_TRI_H
#define USPARSE_KERNELS_LEVEL2_USPARSE_SOLVER_TRI_H

namespace usparse {
  template <typename INDEX_TYPE, typename DATA_TYPE>
  void device_sovlerTri(INDEX_TYPE Am, INDEX_TYPE i, DATA_TYPE *H, DATA_TYPE *s);
}

#endif // USPARSE_KERNELS_LEVEL2_USPARSE_SOLVER_TRI_H
