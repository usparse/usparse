#ifndef COMMON_HIP_CHECK_HPP
#define COMMON_HIP_CHECK_HPP

#include <hip/hip_runtime.h>
#include <iostream>

#define HIP_CHECK(stat)                                                                                                \
  {                                                                                                                    \
    if (stat != hipSuccess) {                                                                                          \
      std::cerr << "Error: " << std::endl;                                                                             \
      std::cerr << "- hip error name: " << hipGetErrorName(stat) << std::endl;                                         \
      std::cerr << "- hip error str: " << hipGetErrorString(stat) << std::endl;                                        \
      std::cerr << "- error in file: " << __FILE__ << std::endl;                                                       \
      std::cerr << "- error in func: " << __FUNCTION__ << std::endl;                                                   \
      std::cerr << "- error in line: " << __LINE__ << std::endl;                                                       \
      exit(-1);                                                                                                        \
    }                                                                                                                  \
  }

#endif // COMMON_HIP_CHECK_HPP
