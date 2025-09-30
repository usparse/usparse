//
// Created by genshen to 2025-07-09.
//

#ifndef USPARSE_KERNELS_LEVEL2_USPARSE_SOLVER_TRI_H
#define USPARSE_KERNELS_LEVEL2_USPARSE_SOLVER_TRI_H

namespace usparse {
  template <typename INDEX_TYPE, typename DATA_TYPE>
  void host_sovlerTri(const INDEX_TYPE Am, const INDEX_TYPE i, const DATA_TYPE *H, DATA_TYPE *s);

  template <typename INDEX_TYPE, typename DATA_TYPE>
  void device_sovlerTri(const INDEX_TYPE Am, const INDEX_TYPE i, const DATA_TYPE *H, DATA_TYPE *s);
} // namespace usparse

#endif // USPARSE_KERNELS_LEVEL2_USPARSE_SOLVER_TRI_H
