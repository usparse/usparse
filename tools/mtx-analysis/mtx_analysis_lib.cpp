#include "init.hpp"
#include "sparseMatrix.hpp"
#include "gmres.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <tuple>
#include <vector>

using RESULT = std::tuple<int, float, double>;
using namespace std;

double _calculateNorm(const double *vec, uint N) {
  double sum = 0.0;
  for (uint i = 0; i < N; ++i) {
    sum += vec[i] * vec[i];
  }
  return std::sqrt(sum);
}

void _spmv(const uint *rowPtr, const uint *colInd, const double *values, const double *x, double *y, uint numRows) {
  for (uint i = 0; i < numRows; ++i) {
    double sum = 0.0;
    for (uint j = rowPtr[i]; j < rowPtr[i + 1]; ++j) {
      sum += values[j] * x[colInd[j]];
    }
    y[i] = sum;
  }
}

// 此函数禁止修改: from demo code
void _initialize(SpM<double> *A, double *x, double *b) {
  int N = A->nrows;

  for (int i = 0; i < N; i++) {
    x[i] = sin(i);
  }

  double beta = _calculateNorm(x, N);
  for (uint i = 0; i < N; i++) {
    x[i] /= beta;
  }

  _spmv(A->rows, A->cols, A->vals, x, b, N);

  for (uint i = 0; i < N; i++)
    x[i] = 0.0;
}

void loadBinToCSR(SpM<double> &A, std::string mtx_path) {
  std::ifstream binFile(mtx_path, std::ios::in | std::ios::binary);

  if (!binFile.is_open()) {
    std::cerr << "can not open output file!" << std::endl;
    return;
  }
  binFile.read(reinterpret_cast<char *>(&A.nrows), sizeof(A.nrows));
  binFile.read(reinterpret_cast<char *>(&A.ncols), sizeof(A.ncols));
  binFile.read(reinterpret_cast<char *>(&A.nnz), sizeof(A.nnz));

  A.rows = (uint *)malloc((A.nrows + 1) * sizeof(uint));
  A.cols = (uint *)malloc(A.nnz * sizeof(uint));
  A.vals = (double *)malloc(A.nnz * sizeof(double));

  binFile.read(reinterpret_cast<char *>(A.rows), (A.nrows + 1) * sizeof(uint));
  binFile.read(reinterpret_cast<char *>(A.cols), A.nnz * sizeof(uint));
  binFile.read(reinterpret_cast<char *>(A.vals), A.nnz * sizeof(double));
  binFile.close();
}

// 幂法估计最大特征值
double estimate_largest_eigenvalue(const SpM<double> &A, int max_iter = 1000, double tol = 1e-6) {
  std::vector<double> x(A.ncols, 1.0), y(A.nrows);
  double lambda_old = 0.0, lambda = 0.0;

  for (int iter = 0; iter < max_iter; ++iter) {
    _spmv(A.rows, A.cols, A.vals, x.data(), y.data(), A.nrows);

    lambda = _calculateNorm(y.data(), A.nrows);
    for (uint i = 0; i < A.nrows; ++i)
      x[i] = y[i] / lambda;

    if (std::abs(lambda - lambda_old) < tol)
      break;
    lambda_old = lambda;
  }

  return lambda;
}

// 反幂法估计最小特征值（适用于 SPD 矩阵）
double estimate_smallest_eigenvalue(const SpM<double> &A, int max_iter = 1000, double tol = 1e-6) {
  std::vector<double> x(A.ncols, 1.0), y(A.nrows), z(A.nrows);
  double lambda_old = 0.0, lambda = 0.0;

  for (int iter = 0; iter < max_iter; ++iter) {
    // 解 Ay = x（简化的雅可比法）
    for (uint i = 0; i < A.nrows; ++i) {
      double diag = 0.0, sum = 0.0;
      for (uint j = A.rows[i]; j < A.rows[i + 1]; ++j) {
        if (A.cols[j] == i)
          diag = A.vals[j];
        else
          sum += A.vals[j] * x[A.cols[j]];
      }
      y[i] = (x[i] - sum) / (diag + 1e-12); // 防止除0
    }

    double norm_y = _calculateNorm(y.data(), A.nrows);
    for (uint i = 0; i < A.nrows; ++i)
      x[i] = y[i] / norm_y;

    _spmv(A.rows, A.cols, A.vals, x.data(), z.data(), A.nrows);
    lambda = 0.0;
    for (uint i = 0; i < A.nrows; ++i)
      lambda += x[i] * z[i];

    if (std::abs(lambda - lambda_old) < tol)
      break;
    lambda_old = lambda;
  }
  return lambda;
}
// --------------------------------------------------------------

int run(int argc, char *argv[]) {
  // ==============禁止修改下列代码==============
  string mtx_name = argv[1];
  mtx_name.erase(mtx_name.begin(), mtx_name.begin() + mtx_name.rfind('/') + 1);
  mtx_name.erase(mtx_name.begin() + mtx_name.find('.'), mtx_name.end());

  SpM<double> A_double;
  loadBinToCSR(A_double, argv[1]);

  std::cout << mtx_name << ": M = " << A_double.nrows << ", N = " << A_double.ncols << std::endl;

  std::vector<double> x_double(A_double.nrows, 0);
  std::vector<double> b_double(A_double.nrows, 1);
  _initialize(&A_double, &x_double[0], &b_double[0]);

  // 条件数分析输出
  double lambda_max = estimate_largest_eigenvalue(A_double);
  double lambda_min = estimate_smallest_eigenvalue(A_double);
  double cond_number = lambda_max / lambda_min;

  std::cout << "Condition Number ≈ " << cond_number;
  if (cond_number > 1e6)
    std::cout << " -> Ill-conditioned matrix (可能病态矩阵)";
  std::cout << std::endl;

  return 0;
}
