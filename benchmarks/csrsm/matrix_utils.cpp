#include <cstring> // 添加 cstring 头文件

#include "matrix_utils.h"
// #include "../include/cli/matrix_market_reader.hpp"
// #include "../include/cli/csr_binary_reader.hpp"
// #include "../include/cli/sparse_format.h"
// 独立矩阵读取函数
template <typename T> MatrixData<T> read_matrix(const char *filepath) {
  MatrixData<T> mat;
  char filename[1000];
  strcpy(filename, filepath);
  csr_mtx<int, T> matrix_in;
  matrix_market_reader<int, T> mm_reader;
  matrix_market<int, T> csr_mm_reader = mm_reader.load_mat(filename);
  matrix_in = csr_mm_reader.to_csr();

  mat.row_ptr = std::vector<int>(matrix_in.row_ptr, matrix_in.row_ptr + (matrix_in.rows + 1));
  mat.col_index = std::vector<int>(matrix_in.col_index, matrix_in.col_index + matrix_in.nnz);
  mat.values = std::vector<T>(matrix_in.values, matrix_in.values + matrix_in.nnz);

  mat.rows = matrix_in.rows;
  mat.cols = matrix_in.cols;
  mat.nnz = matrix_in.nnz;

  // std::cout << "Host Data Created Successfully!\n"
  //           << "=========================================\n"
  //           << "Input Matrix Data Info:\n"
  //           << "rows: " << mat.rows << "\n"
  //           << "cols: " << mat.cols << "\n"
  //           << "nnz:  " << mat.nnz << "\n"
  //           << "=========================================\n";

  return mat;
}

// 修改后的矩阵预处理：提取三角阵并处理对角线元素
template <typename T> MatrixData<T> extract_triangle(const MatrixData<T> &src, bool lower) {
  MatrixData<T> dst;
  dst.rows = src.rows;
  dst.cols = src.cols;
  dst.row_ptr.reserve(src.rows + 1);
  dst.row_ptr.push_back(0);
  dst.valid = true; // 始终标记为有效

  for (int row = 0; row < src.rows; ++row) {
    int start = src.row_ptr[row];
    int end = src.row_ptr[row + 1];
    bool diag_found = false;
    T diag_val = 0;
    int count = 0;

    // 收集满足条件的元素并检查对角线
    for (int i = start; i < end; ++i) {
      int col = src.col_index[i];
      if ((lower && col <= row) || (!lower && col >= row)) {
        // 记录元素
        dst.col_index.push_back(col);
        T val = src.values[i];

        // 检查是否是当前行的对角线
        if (col == row) {
          diag_found = true;
          diag_val = val;
        }
        dst.values.push_back(val);
        count++;
      }
    }

    // 处理对角线元素（新增部分）
    if (!diag_found) {
      // 添加缺失的对角线元素（值为1）
      dst.col_index.push_back(row);
      dst.values.push_back(1.0);
      count++;
    } else if (diag_val == 0) {
      // 找到对角线元素但值为零，修改为1
      // 逆向查找该行最后一个符合条件的位置（优化性能）
      for (int i = dst.col_index.size() - 1; i >= dst.row_ptr.back(); --i) {
        if (dst.col_index[i] == row) {
          dst.values[i] = 1.0;
          break;
        }
      }
    }

    dst.row_ptr.push_back(dst.row_ptr.back() + count);
  }

  dst.nnz = dst.col_index.size();
  return dst;
}

template MatrixData<float> read_matrix<float>(const char *);
template MatrixData<double> read_matrix<double>(const char *);
template MatrixData<float> extract_triangle<float>(const MatrixData<float> &, bool);
template MatrixData<double> extract_triangle<double>(const MatrixData<double> &, bool);