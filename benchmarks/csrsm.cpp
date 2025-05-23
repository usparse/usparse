#include <cstdlib>
#include <iostream>
#include <random>
#include <stdlib.h>
#include <sys/time.h>
#include <vector>

#include "csrsm/common.h"
#include "csrsm/matrix_utils.h"
#include "csrsm/rocsparse_test.h"
#include "csrsm/verification.h"

#include "level3/csrsm_kernels.h"

int main(int argc, char *argv[]) {
  int k = 3;            // 迭代次数
  int rhs = 2;          // 右端项个数
  bool is_lower = true; // 用户自定义上下三角

  // 解析命令行参数

  std::string full_path = "";
  std::string fmt = "mm";
  if (argc != 6) {
    std::cout << "thr num of args can just be 5" << std::endl;
    return -1;
  } else if (fmt != argv[5]) {
    std::cout << "thr file format can just be mm" << std::endl;
    return -1;
  } else {
    k = std::atoi(argv[1]);             // 第一个参数赋值给 k
    rhs = std::atoi(argv[2]) != 0;      // 第二个参数赋值给 rhs
    is_lower = std::atoi(argv[3]) != 0; // 第三个参数赋值给 is_lower
    full_path = argv[4];                // 矩阵路径
  }

  // 读取原始矩阵
  MatrixData<float> orig_matrix_float = read_matrix<float>(full_path.c_str());
  MatrixData<double> orig_matrix_double = read_matrix<double>(full_path.c_str());
  // 预处理得到下(上)三角矩阵
  MatrixData<float> matrix_float = extract_triangle<float>(orig_matrix_float, is_lower);
  MatrixData<double> matrix_double = extract_triangle<double>(orig_matrix_double, is_lower);
  int n = matrix_float.rows;
  // 每行初始化（名、行、列、非零元）
  VerificationResult res;
  res.name = full_path;
  res.rows = matrix_float.rows;
  res.cols = matrix_float.cols;
  res.nnz = matrix_float.nnz;
  // 生成参考b向量、按行（updown kernel）
  // 生成随机数
  std::random_device rd;  // 硬件随机数生成器
  std::mt19937 gen(rd()); // Mersenne Twister 随机数引擎
  // 随机区间1-2
  std::uniform_real_distribution<> dis(0.0, 1.0);

  std::vector<float> h_b_float(n * rhs); // 单精度
  for (int i = 0; i < n; ++i) {
    for (int r = 0; r < rhs; ++r) {
      h_b_float[i * rhs + r] = static_cast<float>(dis(gen));
    }
  }
  std::vector<double> h_b_double(n * rhs); // 双精度
  for (int i = 0; i < n; ++i) {
    for (int r = 0; r < rhs; ++r) {
      h_b_double[i * rhs + r] = static_cast<double>(dis(gen));
    }
  }
  // 按列（基于rocsparse）
  std::vector<float> rh_b_float(n * rhs); // 单精度
  for (int r = 0; r < rhs; ++r) {
    for (int i = 0; i < n; ++i) {
      rh_b_float[r * n + i] = static_cast<float>(h_b_float[i * rhs + r]);
    }
  }
  std::vector<double> rh_b_double(n * rhs); // 双精度
  for (int r = 0; r < rhs; ++r) {
    for (int i = 0; i < n; ++i) {
      rh_b_double[r * n + i] = static_cast<double>(h_b_double[i * rhs + r]);
    }
  }
  // 测试 float 类型的 updown kernel

  int updown_n_float;
  float updown_float_time;
  float *updown_x_float =
      test_updown_matrix<float>(matrix_float, h_b_float.data(), k, rhs, &updown_n_float, is_lower, &updown_float_time);
  res.updown_float = verify_solution<float>(matrix_float, rhs, h_b_float.data(), updown_x_float, false);
  // 将测试数据存入结构体
  res.time.updown_float = updown_float_time;

  // 测试 double 类型的 updown kernel
  int updown_n_double;
  float updown_double_time;
  double *updown_x_double = test_updown_matrix<double>(matrix_double, h_b_double.data(), k, rhs, &updown_n_double,
                                                       is_lower, &updown_double_time);
  res.updown_double = verify_solution<double>(matrix_double, rhs, h_b_double.data(), updown_x_double, false);
  // 将测试数据存入结构体
  res.time.updown_double = updown_double_time;

  // 测试 float 类型的 rocsparse
  int m_rocsparse_float;
  float rocsparse_float_time;
  float *rocsparse_x_float =
      test_rocsparse_matrix<float>(matrix_float, rh_b_float.data(), k, rhs, &m_rocsparse_float, &rocsparse_float_time);
  res.rocsparse_float = verify_solution<float>(matrix_float, rhs, rh_b_float.data(), rocsparse_x_float, true);
  // 将测试数据存入结构体
  res.time.rocsparse_float = rocsparse_float_time;

  // 测试 double 类型的 rocsparse
  int m_rocsparse_double;
  float rocsparse_double_time;
  double *rocsparse_x_double = test_rocsparse_matrix<double>(matrix_double, rh_b_double.data(), k, rhs,
                                                             &m_rocsparse_double, &rocsparse_double_time);
  res.rocsparse_double = verify_solution<double>(matrix_double, rhs, rh_b_double.data(), rocsparse_x_double, true);
  // 将测试数据存入结构体
  res.time.rocsparse_double = rocsparse_double_time;

  // 清理资源
  delete[] updown_x_float;
  delete[] updown_x_double;
  delete[] rocsparse_x_float;
  delete[] rocsparse_x_double;

  // 写入CSV文件
  write_verification_csv(res);

  return 0;
}