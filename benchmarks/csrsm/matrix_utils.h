#pragma once
#include "common.h"

template <typename T> MatrixData<T> read_matrix(const char *filepath);

template <typename T> MatrixData<T> extract_triangle(const MatrixData<T> &src, bool lower);
