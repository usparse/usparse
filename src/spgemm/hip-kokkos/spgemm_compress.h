#ifndef SPGEMM_KOKKOS_SPGEMM_COMPRESS_H
#define SPGEMM_KOKKOS_SPGEMM_COMPRESS_H

#include "compress_matrix.h"
#include "spgemm_compress_impl.hpp"
/**
 * compress matrix
 * matrix: input, csr format matrix
 * return: output, compress format matrix
 **/
template <typename T, typename I> CompressMatrix<I> spgemm_compress(const CsrMatrix<T, I> &matrix);

#endif // SPGEMM_KOKKOS_SPGEMM_COMPRESS_H