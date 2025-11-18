
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

#include "usparse_matrix.h"

#include "gmres.hpp"
#include "init.hpp"
#include "sparseMatrix.hpp"
#include "solver/gmres/kernels/usparse_gmres.h"

using RESULT = std::tuple<int, float, double>;

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

int main(int argc, char *argv[]) {
  // ==============禁止修改下列代码==============
  string mtx_name = argv[1];
  mtx_name.erase(mtx_name.begin(), mtx_name.begin() + mtx_name.rfind('/') + 1);
  mtx_name.erase(mtx_name.begin() + mtx_name.find('.'), mtx_name.end());

  SpM<double> A_double;
  loadBinToCSR(A_double, argv[1]);

  std::cout << mtx_name << ": M = " << A_double.nrows << ", N = " << A_double.ncols << std::endl;

  std::vector<double> x_double(A_double.nrows, 0);
  std::vector<double> b_double(A_double.nrows, 1);
  initialize(&A_double, &x_double[0], &b_double[0]);

  // init device
  initDCU(0);

  /**
   * initialize deivce data
   */
  SpM<double> dev_A;
  double *dev_x = nullptr;
  double *dev_y = nullptr; // result = A * X

  var_csr_desc<int, double> d_var_csr_desc;
  init_device_data(&A_double, &x_double[0], &dev_A, &dev_x, &dev_y, &d_var_csr_desc); // 申请内存
  auto d_csr_desc = d_var_csr_desc.as_const();
  // ==============禁止修改上列代码==============

  // warming up start
  gmres_float(&A_double, &x_double[0], &b_double[0], &dev_A, dev_x, dev_y, d_csr_desc);
  HIP_CHECK(hipMemset(dev_x, 0, A_double.nrows * sizeof(double))); // reset the solve result.
  // warming up end.

  auto res = gmres_float(&A_double, &x_double[0], &b_double[0], &dev_A, dev_x, dev_y, d_csr_desc);
  // auto res = gmres(&A_double, &x_double[0], &b_double[0], &dev_A, dev_x, dev_y,
  //                  d_csr_desc); // GMRES算法核心，需重点优化

  // destory data
  destory_device_data(&dev_A, dev_y, dev_x);

  // ==============禁止修改下列结果输出==============
  std::cout << "iters = " << std::get<0>(res) << ", time = " << std::get<1>(res) << "ms, resid = " << std::get<2>(res)
            << std::endl;

  std::ofstream outfile("gmres_time.txt", std::ios::app);
  if (!outfile.is_open()) {
    std::cerr << "can not open output file!" << std::endl;
    return 1;
  }
  outfile << mtx_name << " " << std::get<0>(res) << " " << std::get<1>(res) << " " << std::get<2>(res) << "\n";
  outfile.close();
  // ==============禁止修改上列结果输出==============
  return 0;
}
