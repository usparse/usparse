//
// Created by genshen to 2025-08-28.
//

#ifndef USPARSE_KERNELS_USPARSE_SPMV_API_CONFIG_HPP
#define USPARSE_KERNELS_USPARSE_SPMV_API_CONFIG_HPP


constexpr int BETA_Y_POLICY_AUTO = 0; // does not guarantee the value of beat in y = alpha * Ax + beta * y.
constexpr int BETA_Y_POLICY_ZERO = 1; // beta = 0.0 in y = alpha * Ax + beta * y.
constexpr int BETA_Y_POLICY_ONE = 2;  // beta = 1.0 in y = alpha * Ax + beta * y.
constexpr int BETA_Y_POLICY_OTHER = 3; // other case.

#endif // USPARSE_KERNELS_USPARSE_SPMV_API_CONFIG_HPP
