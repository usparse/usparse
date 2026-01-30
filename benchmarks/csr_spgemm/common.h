//
// Created by luxu on 2022/12/08.
//

#ifndef SPGEMM_BENCHMARK_COMMON_H
#define SPGEMM_BENCHMARK_COMMON_H

#ifdef DEVICE_SIDE_VERIFY_FLAG
#include <hipsparse.h>
#include <rocsparse.h>
#endif

#include "common/usparse_utils.hpp"
#include "spgemm_common/timer/timer.hpp"

#define ROCSPARSE_CHECK(stat)                                                                                          \
  {                                                                                                                    \
    if (stat != rocsparse_status_success) {                                                                            \
      std::cerr << "Error: rocsparse error " << stat << " in line " << __LINE__ << std::endl;                          \
      exit(-1);                                                                                                        \
    }                                                                                                                  \
  }

#endif // SPGEMM_BENCHMARK_COMMON_H
