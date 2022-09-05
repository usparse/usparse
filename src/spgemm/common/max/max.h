#ifndef COMMON_MAX_H
#define COMMON_MAX_H

#include "max_impl.hpp"

template <typename T> size_t max_buffer_size(const size_t input_size);

template <typename T> void max(const size_t input_size, const T *input, T *max_num, void *temp_buffer);

template <typename T> void max(const size_t input_size, const T *input, T *max_num);

#endif // COMMON_MAX_H