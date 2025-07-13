//
// Created by genshen to 2025-07-09.
//

#ifndef GTEST_TESTS_UTILS_HPP
#define GTEST_TESTS_UTILS_HPP

#define TEST_HIP_CHECK(stat)                                                                                           \
  {                                                                                                                    \
    hipError_t err = stat;                                                                                             \
    if (err != hipSuccess) {                                                                                           \
      std::cerr << "Error: HIP reports " << hipGetErrorString(err) << std::endl;                                       \
      std::cerr << "Raised in file " << __FILE__ << "#" << __func__ << "#line " << __LINE__ << std::endl;              \
      std::abort();                                                                                                    \
    }                                                                                                                  \
  }

#endif // GTEST_TESTS_UTILS_HPP
