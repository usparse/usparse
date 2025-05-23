#pragma once
#include "common.h"

template <typename T>
VerificationResultBase verify_solution(const MatrixData<T> &matrix, int rhs, const T *b_gpu, const T *x_gpu,
                                       bool col_p);

template <typename T> std::string format_value(T value, bool force_scientific);

// void write_verification_csv(
//     const std::string& filename,
//     const std::vector<VerificationResult>& results);
void write_verification_csv(const VerificationResult &results);
