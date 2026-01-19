//
// Created by chu genshen on 2021/9/1.
//

#ifndef USPARSE_HIP_USPARSE_FLAT_CONFIG_H
#define USPARSE_HIP_USPARSE_FLAT_CONFIG_H

constexpr int FLAT_REDUCE_OPTION_VEC = 0;
constexpr int FLAT_REDUCE_OPTION_VEC_MEM_COALESCING = 1;
constexpr int FLAT_REDUCE_OPTION_DIRECT = 2;
constexpr int FLAT_REDUCE_OPTION_SEGMENT_SUM = 3;

constexpr int DEFAULT_FLAT_REDUCE_OPTION = FLAT_REDUCE_OPTION_VEC_MEM_COALESCING;

constexpr bool FLAT_ONE_PASS = true;
constexpr bool FLAT_ONE_PASS_ADAPTIVE = true;

// undeterministic csrmv algorithm with atomicAdd. It may be faster than the deterministic method..
constexpr int FLAT_STORE_Y_POLICY_UNDETERMINISTIC = 0;
// deterministic csrmv algorithm with atomicAdd. It is not as fast as the undeterministic method.
constexpr int FLAT_STORE_Y_POLICY_DETERMINISTIC = 1; 

#endif // USPARSE_HIP_USPARSE_FLAT_CONFIG_H
