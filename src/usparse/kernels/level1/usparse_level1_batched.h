//
// Created by renyi to 2025-09-16.
//

#ifndef USPARSE_KERNELS_LEVEL1_USPARSE_LEVEL1_BATCHED_H
#define USPARSE_KERNELS_LEVEL1_USPARSE_LEVEL1_BATCHED_H

namespace usparse {
  template <typename INDEX_TYPE, typename DATA_TYPE>
  void gs_dotproduct(const DATA_TYPE *d_r0, const DATA_TYPE *d_v, DATA_TYPE *d_h, DATA_TYPE *d_temp_buffer,
                     const INDEX_TYPE N, const INDEX_TYPE i, std::size_t *temp_buffer_size_byte);

  template <typename INDEX_TYPE, typename DATA_TYPE>
  void gs_daxpy(DATA_TYPE *d_r0, const DATA_TYPE *d_v, const DATA_TYPE *d_h, const INDEX_TYPE N, const INDEX_TYPE i);
} // namespace usparse

#endif // USPARSE_KERNELS_LEVEL1_USPARSE_LEVEL1_BATCHED_H
