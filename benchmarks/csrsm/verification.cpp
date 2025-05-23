#include "verification.h"
#include <cmath>
#include <fstream>
#include <iostream>

// 模板化正确性验证b函数
template <typename T>
VerificationResultBase verify_solution(const MatrixData<T> &matrix, int rhs, const T *b_gpu, const T *x_gpu,
                                       bool col_p) {
  int n = matrix.rows;
  VerificationResultBase result{};
  const int total_elements = matrix.rows * rhs;
  // 总相对误差及最大误差
  int rel_err_sum1e5 = 0;  // 单精
  int rel_err_sum1e12 = 0; // 双精
  double max_rel_sum = 0.0;
  double max_abs_sum = 0.0;
  // 对每个右端向量进行验证
  for (int r = 0; r < rhs; ++r) {
    // 初始化b
    std::vector<T> b_cpu(n, 0.0);
    // 计算Ax
    for (int row = 0; row < n; ++row) {
      T sum = 0.0;
      int start = matrix.row_ptr[row];
      int end = matrix.row_ptr[row + 1];
      for (int j = start; j < end; ++j) {
        int col = matrix.col_index[j];
        // 区分行优先还是列优先
        if (!col_p)
          sum += matrix.values[j] * x_gpu[col * rhs + r];
        else
          sum += matrix.values[j] * x_gpu[r * n + col];
      }
      b_cpu[row] = sum;
    }

    // 统计误差
    int abs_err_1e5 = 0, rel_err_1e5 = 0;
    int abs_err_1e12 = 0, rel_err_1e12 = 0;
    double max_abs = 0.0, max_rel = 0.0;

    for (int i = 0; i < n; ++i) {
      // 区分行优先还是列优先
      double expected = (!col_p) ? b_gpu[i * rhs + r] : b_gpu[r * n + i];
      double actual = b_cpu[i];
      double abs_err = std::abs(actual - expected);
      double rel_err = (expected != 0) ? abs_err / std::abs(expected) : abs_err;

      // 更新最大误差
      if (abs_err > max_abs)
        max_abs = abs_err;
      if (rel_err > max_rel)
        max_rel = rel_err;

      // 统计阈值
      if (abs_err > 1e-5)
        abs_err_1e5++;
      if (rel_err > 1e-5)
        rel_err_1e5++;
      if (abs_err > 1e-12)
        abs_err_1e12++;
      if (rel_err > 1e-12)
        rel_err_1e12++;
    }

    // 总计
    if (max_rel > max_rel_sum)
      max_rel_sum = max_rel;
    if (max_abs > max_abs_sum)
      max_abs_sum = max_abs;

    rel_err_sum1e5 += rel_err_1e5;
    rel_err_sum1e12 += rel_err_1e12;
  }
  // std::cout << "Max absolute error: " << max_abs_sum << std::endl;
  // std::cout << "Max relative error: " << max_rel_sum << std::endl;
  // std::cout << "Elements with relative error > 1e-5: " << rel_err_sum1e5 << "/" << total_elements << std::endl;
  // std::cout << "Elements with relative error > 1e-12: " << rel_err_sum1e12 << "/" << total_elements<< std::endl;
  result.max_abs_error = static_cast<double>(max_abs_sum);
  result.max_rel_error = static_cast<double>(max_rel_sum);
  result.count_rel_1e5 = rel_err_sum1e5;
  result.count_rel_1e12 = rel_err_sum1e12;
  return result;
}

// 数值格式化函数 (verify)
template <typename T> std::string format_value(T value, bool force_scientific) {
  std::ostringstream oss;
  if (force_scientific || (std::abs(value) > 1e4 || (value != 0 && std::abs(value) < 1e-4))) {
    oss << std::scientific << std::setprecision(6);
  } else {
    oss << std::fixed << std::setprecision(6);
  }
  oss << value;
  return oss.str();
}

// CSV写入函数 (verify)
// void write_verification_csv(const std::string& filename,
//                            const std::vector<VerificationResult>& results) {
//     std::ofstream file(filename);
//     if (!file.is_open()) {
//         std::cerr << "Error opening file: " << filename << std::endl;
//         return;
//     }

//     // 表头
//     file << "Matrix,Rows,Cols,NNZ,"
//          << "UpdownFloat_Time(us),UF_MaxAbs,UF_MaxRel,UF_>1e5,UF_>1e12,"
//          << "UpdownDouble_Time(us),UD_MaxAbs,UD_MaxRel,UD_>1e5,UD_>1e12,"
//          << "RocsparseFloat_Time(us),RF_MaxAbs,RF_MaxRel,RF_>1e5,RF_>1e12,"
//          << "RocsparseDouble_Time(us),RD_MaxAbs,RD_MaxRel,RD_>1e5,RD_>1e12\n";

//     // 数据行
//     for (const auto& res : results) {
//         file << res.name << ","
//              << res.rows << "," << res.cols << "," << res.nnz << ","
//              << format_value(res.time.updown_float) << ","
//              << format_value(res.updown_float.max_abs) << ","
//              << format_value(res.updown_float.max_rel) << ","
//              << res.updown_float.count_1e5 << "," << res.updown_float.count_1e12 << ","
//              << format_value(res.time.updown_double) << ","
//              << format_value(res.updown_double.max_abs) << ","
//              << format_value(res.updown_double.max_rel) << ","
//              << res.updown_double.count_1e5 << "," << res.updown_double.count_1e12 << ","
//              << format_value(res.time.rocsparse_float) << ","
//              << format_value(res.rocsparse_float.max_abs) << ","
//              << format_value(res.rocsparse_float.max_rel) << ","
//              << res.rocsparse_float.count_1e5 << "," << res.rocsparse_float.count_1e12 << ","
//              << format_value(res.time.rocsparse_double) << ","
//              << format_value(res.rocsparse_double.max_abs) << ","
//              << format_value(res.rocsparse_double.max_rel) << ","
//              << res.rocsparse_double.count_1e5 << "," << res.rocsparse_double.count_1e12 << "\n";
//     }
// }

void write_verification_csv(const VerificationResult &results) {
  // 表头
  std::cout << "Matrix,Rows,Cols,NNZ,"
            << "UpdownFloat_Time(us),UF_MaxAbs,UF_MaxRel,UF_>1e5,UF_>1e12,"
            << "UpdownDouble_Time(us),UD_MaxAbs,UD_MaxRel,UD_>1e5,UD_>1e12,"
            << "RocsparseFloat_Time(us),RF_MaxAbs,RF_MaxRel,RF_>1e5,RF_>1e12,"
            << "RocsparseDouble_Time(us),RD_MaxAbs,RD_MaxRel,RD_>1e5,RD_>1e12" << std::endl;
  // 数据行

  std::cout << results.name << "," << results.rows << "," << results.cols << "," << results.nnz << ","
            << format_value(results.time.updown_float) << "," << format_value(results.updown_float.max_abs) << ","
            << format_value(results.updown_float.max_rel) << "," << results.updown_float.count_1e5 << ","
            << results.updown_float.count_1e12 << "," << format_value(results.time.updown_double) << ","
            << format_value(results.updown_double.max_abs) << "," << format_value(results.updown_double.max_rel) << ","
            << results.updown_double.count_1e5 << "," << results.updown_double.count_1e12 << ","
            << format_value(results.time.rocsparse_float) << "," << format_value(results.rocsparse_float.max_abs) << ","
            << format_value(results.rocsparse_float.max_rel) << "," << results.rocsparse_float.count_1e5 << ","
            << results.rocsparse_float.count_1e12 << "," << format_value(results.time.rocsparse_double) << ","
            << format_value(results.rocsparse_double.max_abs) << "," << format_value(results.rocsparse_double.max_rel)
            << "," << results.rocsparse_double.count_1e5 << "," << results.rocsparse_double.count_1e12 << std::endl;
}

template std::string format_value<float>(float value, bool force_scientific);
template std::string format_value<double>(double value, bool force_scientific);
template VerificationResultBase verify_solution<float>(const MatrixData<float> &matrix, int rhs, const float *b_gpu,
                                                       const float *x_gpu, bool col_p);
template VerificationResultBase verify_solution<double>(const MatrixData<double> &matrix, int rhs, const double *b_gpu,
                                                        const double *x_gpu, bool col_p);
