#ifndef COMMON_SCAN_IMPL_HPP
#define COMMON_SCAN_IMPL_HPP

#include "../check/hip_check.hpp"
#include "lookback_scan.hpp"
#include "reduce_then_scan.hpp"
#include "single_block_scan.hpp"

namespace common {
template <typename T> size_t scan_buffer_size(const size_t input_size) {
  if (input_size <= single_block_scan::THREADS) {
    return single_block_scan::single_block_scan_buffer_size<T>(input_size);
  } else {
    return lookback_scan::decoupled_look_back_scan_buffer_size<T>(input_size);
  }
}

template <typename T> void scan(const size_t input_size, T *input, void *temp_buffer) {
  if (input_size <= single_block_scan::THREADS) {
    single_block_scan::single_block_scan_impl<T>(input_size, input, input);
  } else {
    lookback_scan::decoupled_look_back_scan_impl<T>(input_size, input, input, temp_buffer);
  }
}

template <typename T> void scan(const size_t input_size, const T *input, T *output, void *temp_buffer) {
  if (input_size <= single_block_scan::THREADS) {
    single_block_scan::single_block_scan_impl<T>(input_size, input, output);
  } else {
    lookback_scan::decoupled_look_back_scan_impl<T>(input_size, input, output, temp_buffer);
  }
}
} // namespace common
#endif // COMMON_SCAN_IMPL_HPP
