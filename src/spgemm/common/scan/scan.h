#ifndef COMMON_SCAN_H
#define COMMON_SCAN_H

#include "scan_impl.hpp"

template <typename T> size_t scan_buffer_size(const size_t input_size);

template <typename T> void scan(const size_t input_size, T *input, void *temp_buffer);

template <typename T> void scan(const size_t input_size, const T *input, T *output, void *temp_buffer);

#endif // COMMON_SCAN_H