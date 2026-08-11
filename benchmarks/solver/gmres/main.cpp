
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

#include "usparse_matrix.h"

#include "gmres.hpp"
#include "init.hpp"
#include "sparseMatrix.hpp"
#include "solver/gmres/kernels/usparse_gmres.h"


#include "../../utils/benchmark_main.hpp"

UBENCH_MAIN(ubench::BENCH_FORMAT_CSR);

void bench_gmres(const std::string mtx_path, csr_mtx<type_sparse_default_index, usparse::usparse_fp64> h_csr);

template <typename PRECISION>
int run_csr_benchmark(std::string mtx_path, csr_mtx<type_sparse_default_index, PRECISION> h_csr) {
  // if precision is double, run gmres_double, else return with error:
  if constexpr (std::is_same<PRECISION, double>::value) {
    bench_gmres(mtx_path, h_csr);
  } else {
    std::cerr << "error: unsupported precision type. Only double is supported." << std::endl;
    return -1;
  }
  return 0;
}

using RESULT = std::tuple<int, float, double>;

void bench_gmres(const std::string mtx_path, csr_mtx<type_sparse_default_index, usparse::usparse_fp64> h_csr) {
  SpM<double> A_double;
  A_double.nrows = h_csr.rows;
  A_double.ncols = h_csr.cols;
  A_double.nnz = h_csr.nnz;
  A_double.rows = reinterpret_cast<uint *>(h_csr.row_ptr); // int * -> unint * // todo:
  A_double.cols = reinterpret_cast<uint *>(h_csr.col_index); // int * -> unint * // todo:
  A_double.vals = h_csr.values;

  std::cout << mtx_path << ": M = " << A_double.nrows << ", N = " << A_double.ncols << std::endl;

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
  USPARSE_HIP_API_CHECK(hipMemset(dev_x, 0, A_double.nrows * sizeof(double))); // reset the solve result.
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
    return;
  }
  outfile << mtx_path << " " << std::get<0>(res) << " " << std::get<1>(res) << " " << std::get<2>(res) << "\n";
  outfile.close();
  // ==============禁止修改上列结果输出==============
}
