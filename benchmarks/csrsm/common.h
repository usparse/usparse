#pragma once
#include <algorithm>
#include <cstring>
#include <fstream>
#include <hip/hip_runtime.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <sys/time.h>
#include <vector>

#include "csr_binary_reader.hpp"
#include "matrix_market_reader.hpp"
#include "sparse_format.h"

#include "level3/csrsm_kernels.h"

#define WARP_SIZE 64

// --------------------- 验证结果结构 ---------------------
struct VerificationResultBase {
  double max_abs_error;
  double max_rel_error;
  int count_rel_1e5;
  int count_rel_1e12;
};

struct VerificationResult {
  // 矩阵基本信息
  std::string name;
  int rows;
  int cols;
  int nnz;

  // 时间统计（毫秒）
  struct TimeStats {
    double updown_float = 0.0;
    double updown_double = 0.0;
    double rocsparse_float = 0.0;
    double rocsparse_double = 0.0;
  } time;

  // 精度统计
  struct AccuracyStats {
    double max_abs = 0.0;
    double max_rel = 0.0;
    int count_1e5 = 0;
    int count_1e12 = 0;

    AccuracyStats &operator=(const VerificationResultBase &base) {
      max_abs = base.max_abs_error;
      max_rel = base.max_rel_error;
      count_1e5 = base.count_rel_1e5;
      count_1e12 = base.count_rel_1e12;
      return *this;
    }
  };

  AccuracyStats updown_float;
  AccuracyStats updown_double;
  AccuracyStats rocsparse_float;
  AccuracyStats rocsparse_double;
};

// 统计耗时
struct my_timer {
  struct timeval start_time, end_time;
  double time_use;

  void start() { gettimeofday(&start_time, NULL); }

  void stop() {
    gettimeofday(&end_time, NULL);
    time_use = (end_time.tv_sec - start_time.tv_sec) * 1.0e6 + end_time.tv_usec - start_time.tv_usec;
  }
};

// 数值格式化函数 (verify)
template <typename T> std::string format_value(T value, bool force_scientific = false);
