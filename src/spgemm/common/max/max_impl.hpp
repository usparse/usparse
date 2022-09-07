#ifndef COMMON_MAX_IMPL_HPP
#define COMMON_MAX_IMPL_HPP

#include "block_reduce_max.hpp"

namespace common {
template <typename T> size_t max_buffer_size(const size_t input_size) {
  return block_reduce_max_buffer_size<T>(input_size);
}

template <typename T> void max(const size_t input_size, const T *input, T *max_num, void *temp_buffer) {
  block_reduce_max_impl<T>(input_size, input, max_num, temp_buffer);
}

template <typename T> void max(const size_t input_size, const T *input, T *max_num) {
  block_reduce_max_wrapper<T>(input_size, input, max_num);
}
} // namespace common

#endif // COMMON_MAX_IMPL_HPP