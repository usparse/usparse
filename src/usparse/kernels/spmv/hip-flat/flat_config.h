//
// Created by chu genshen on 2021/9/1.
//

#ifndef SPMV_ACC_FLAT_CONFIG_H
#define SPMV_ACC_FLAT_CONFIG_H

constexpr int FLAT_REDUCE_OPTION_VEC = 0;
constexpr int FLAT_REDUCE_OPTION_VEC_MEM_COALESCING = 1;
constexpr int FLAT_REDUCE_OPTION_DIRECT = 2;

constexpr int DEFAULT_FLAT_REDUCE_OPTION = FLAT_REDUCE_OPTION_VEC_MEM_COALESCING;

constexpr bool FLAT_ONE_PASS = true;
constexpr bool FLAT_ONE_PASS_ADAPTIVE = true;

constexpr int BEAT_Y_POLICY_AUTO = 0; // does not guarantee the value of beat in y = alpha * Ax + beta * y.
constexpr int BEAT_Y_POLICY_ZERO = 1; // beta = 0.0 in y = alpha * Ax + beta * y.
constexpr int BEAT_Y_POLICY_ONE = 2;  // beta = 1.0 in y = alpha * Ax + beta * y.
constexpr int BEAT_Y_POLICY_OTHER = 3; // other case.

#endif // SPMV_ACC_FLAT_CONFIG_H
