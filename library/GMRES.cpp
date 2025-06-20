// gmres_restarted_manual.cpp
// C++ implementation of restarted GMRES(m) without Eigen
// Uses only standard C++ STL and raw arrays
// For illustration purposes; not optimized for production use

#include <cassert>
#include <cmath>
#include <iostream>
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

// Matrix-vector multiplication
Vector matvec(const Matrix &A, const Vector &x) {
  size_t n = A.size();
  Vector result(n, 0.0);
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < x.size(); ++j)
      result[i] += A[i][j] * x[j];
  return result;
}

// 2-norm
double norm(const Vector &v) { return std::sqrt(dot(v, v)); }

// GMRES(m) without preconditioning
double gmres(const Matrix &A, const Vector &b, Vector &x, int m, int maxIters, double tol) {
  int n = A.size();
  Vector r(n);
  Vector r_new(n);
  std::vector<Vector> V(m + 1, Vector(n));
  std::vector<std::vector<double>> H(m + 1, std::vector<double>(m, 0.0));
  std::vector<double> cs(m, 0.0), sn(m, 0.0), g(m + 1, 0.0);

  int iter = 0;
  while (iter < maxIters) {
    Vector Ax = matvec(A, x);
    for (int i = 0; i < n; ++i)
      r[i] = b[i] - Ax[i];
    double beta = norm(r);
    if (beta < tol) {
      printf("## return!\n");
      return beta;
    }

    for (int i = 0; i < n; ++i)
      V[0][i] = r[i] / beta;
    std::fill(g.begin(), g.end(), 0.0);
    g[0] = beta;

    int j;
    for (j = 0; j < m && iter < maxIters; ++j, ++iter) {
      Vector w = matvec(A, V[j]);
      // for(int i = 0; i< n; i++){
      //    std::cout<<Ax[i]<<" "<<std::endl;
      // }

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

    r_new = matvec(A, x);
    for (int i = 0; i < n; ++i)
      r_new[i] = b[i] - r_new[i];
    // if (norm(r_new) < tol)
    //     return true;
  }

  return norm(r_new);
}

int main() {
  int n = 4;
  Matrix A(n, Vector(n, 0.0));
  A[0][0] = 1.0;
  A[1][1] = 2.0;
  A[2][2] = 5.0;
  A[3][3] = 4.0;

  Vector b(n, 1.0);
  Vector x(n, 0.0);

  int m = 30;
  int maxIters = 200;
  double tol = 1e-4;

  double norm = gmres(A, b, x, m, maxIters, tol);

  if (norm < tol) {
    std::cout << "GMRES converged. x = [ ";
    for (double xi : x)
      std::cout << xi << " ";
    std::cout << "]\n";
    std::cout << "nrom = " << norm << std::endl;
  } else {
    std::cout << "GMRES did not converge.\n";
  }

  return 0;
}
