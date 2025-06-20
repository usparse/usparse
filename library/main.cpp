#include "dtype.h"
#include "gmres_crs.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <string.h>
#include <vector>

int main() {
  auto csr_mtx = load_mat<int, double>("./rdb968.bin2");
  int row = csr_mtx.rows;
  // int n = 5;
  // Matrix A(n, Vector(n, 0.0));
  // for (int i = 0; i < n; ++i) {
  //     A[i][i] = 4.0;
  //     if (i > 0) A[i][i - 1] = -1.0;
  //     if (i < n - 1) A[i][i + 1] = -1.0;
  // }

  /**
   * Demo test
   */
  // int row = 4;
  // int col = 4;
  // int nnz = 4;
  // double val[4] = {1, 2, 5, 4};
  // int col_ptr[4] = {0,1,2,3};
  // int row_ptr[5] = {0,1,2,3,4};
  // int row = csr_mtx.rows;

  Vector b(row, 1.0);
  Vector x(row, 0.0);

  int m = 30;
  int maxIters = 200;
  double tol = 1e-4;

  // double r_new = gmres(val, row_ptr, col_ptr, row, col, b, x, m, maxIters, tol);
  double r_new =
      gmres(csr_mtx.values, csr_mtx.row_ptr, csr_mtx.col_index, csr_mtx.rows, csr_mtx.cols, b, x, m, maxIters, tol);

  if (r_new < tol) {
    std::cout << "GMRES converged. x = [ ";
    for (int i = 0; i < row; i++) {
      std::cout << x[i] << " ";
    }
    std::cout << "]\n";
    std::cout << "nrom = " << r_new << std::endl;
  } else {
    std::cout << "GMRES did not converge.\n";
  }

  return 0;
}
