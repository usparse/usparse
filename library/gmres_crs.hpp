// gmres_restarted_manual.cpp
// C++ implementation of restarted GMRES(m) without Eigen
// Uses only standard C++ STL and raw arrays
// For illustration purposes; not optimized for production use

#include "dtype.h"
#include "read_crs.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <string.h>
#include <vector>

using Vector = std::vector<double>;
using Matrix = std::vector<std::vector<double>>;

// Dot product
double dot(const Vector &a, const Vector &b) {
  double sum = 0.0;
  for (size_t i = 0; i < a.size(); ++i)
    sum += a[i] * b[i];
  return sum;
}

// Scalar-vector multiplication and addition
void axpy(double alpha, const Vector &x, Vector &y) {
  for (size_t i = 0; i < x.size(); ++i)
    y[i] += alpha * x[i];
}

// computer y = A*x
// void device_spmv(const dtype *dev_value, const int *dev_rowptr, const int *dev_colindex, int m, int n, const double
// *dev_x, double *dev_y) {

// }
/** compute y = Ax
 * @param value: crs格式中非零元素的值
 * @param rowptr: 行坐标信息
 * @param colindex: 列坐标信息
 * @param m: A的行大小
 * @param n: A的列大小
 * @param x:
 * @return result(m,1)
 */
Vector host_spmv(const dtype *value, const int *rowptr, const int *colindex, int m, int n, const Vector &x) {
  // calculate the matrix-vector multiply where matrix is stored in the form of CSR
  Vector result(m, 0.0);
  for (int i = 0; i < m; i++) {
    dtype y0 = 0;
    for (int j = rowptr[i]; j < rowptr[i + 1]; j++) {
      y0 += value[j] * x[colindex[j]];
    }
    result[i] = y0 + result[i];
  }
  return result;
}

// 2-norm
double norm(const Vector &v) { return std::sqrt(dot(v, v)); }

// GMRES(m) without preconditioning
double gmres(const dtype *value, const int *rowptr, const int *colindex, int A_m, int A_n, const Vector &b, Vector &x,
             int m, int maxIters, double tol) {
  int n = A_m;
  Vector r(n);
  Vector r_new(n);
  std::vector<Vector> V(m + 1, Vector(n));
  std::vector<std::vector<double>> H(m + 1, std::vector<double>(m, 0.0));
  std::vector<double> cs(m, 0.0), sn(m, 0.0), g(m + 1, 0.0);

  int iter = 0;
  while (iter < maxIters) {
    /**
     * 向量矩阵乘
     */
    Vector Ax = host_spmv(value, rowptr, colindex, A_m, A_n, x);
    for (int i = 0; i < n; ++i)
      r[i] = b[i] - Ax[i];
    double beta = norm(r);
    if (beta < tol) {
      return beta;
    }

    for (int i = 0; i < n; ++i)
      V[0][i] = r[i] / beta;
    std::fill(g.begin(), g.end(), 0.0);
    g[0] = beta;

    int j;
    for (j = 0; j < m && iter < maxIters; ++j, ++iter) {
      Vector w = host_spmv(value, rowptr, colindex, A_m, A_n, V[j]);
      for (int i = 0; i <= j; ++i) {
        H[i][j] = dot(V[i], w);
        for (int k = 0; k < n; ++k)
          w[k] -= H[i][j] * V[i][k];
      }
      H[j + 1][j] = norm(w);
      if (H[j + 1][j] != 0.0)
        for (int k = 0; k < n; ++k)
          V[j + 1][k] = w[k] / H[j + 1][j];

      for (int i = 0; i < j; ++i) {
        double temp = cs[i] * H[i][j] + sn[i] * H[i + 1][j];
        H[i + 1][j] = -sn[i] * H[i][j] + cs[i] * H[i + 1][j];
        H[i][j] = temp;
      }

      double delta = std::hypot(H[j][j], H[j + 1][j]);
      cs[j] = H[j][j] / delta;
      sn[j] = H[j + 1][j] / delta;
      H[j][j] = cs[j] * H[j][j] + sn[j] * H[j + 1][j];
      H[j + 1][j] = 0.0;

      double temp = cs[j] * g[j] + sn[j] * g[j + 1];
      g[j + 1] = -sn[j] * g[j] + cs[j] * g[j + 1];
      g[j] = temp;

      if (std::abs(g[j + 1]) < tol) {
        ++j;
        break;
      }
    }

    std::vector<double> y(j, 0.0);
    for (int i = j - 1; i >= 0; --i) {
      y[i] = g[i];
      for (int k = i + 1; k < j; ++k)
        y[i] -= H[i][k] * y[k];
      y[i] /= H[i][i];
    }

    for (int i = 0; i < j; ++i)
      axpy(y[i], V[i], x);

    r_new = host_spmv(value, rowptr, colindex, A_m, A_n, x);
    for (int i = 0; i < n; ++i)
      r_new[i] = b[i] - r_new[i];
    // if (norm(r_new) < tol)
    //     return true;
  }

  return norm(r_new);
}
