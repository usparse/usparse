
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

#include "usparse_matrix.h"

#include "gmres.hpp"
#include "init.hpp"
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
  usparse::csr_mat_descr<double, int> h_csr_desc = h_csr;
  // A_double.nrows = h_csr.rows;
  // A_double.ncols = h_csr.cols;
  // A_double.nnz = h_csr.nnz;
  // A_double.rows = reinterpret_cast<uint *>(h_csr.row_ptr); // int * -> unint * // todo:
  // A_double.cols = reinterpret_cast<uint *>(h_csr.col_index); // int * -> unint * // todo:
  // A_double.vals = h_csr.values;

  std::cout << mtx_path << ": M = " << h_csr_desc.rows << ", N = " << h_csr_desc.cols << std::endl;

  std::vector<double> x_double(h_csr_desc.rows, 0);
  std::vector<double> b_double(h_csr_desc.rows, 1);
  initialize(h_csr_desc, &x_double[0], &b_double[0]);

  // init device
  initDCU(0);

  /**
   * initialize deivce data
   */
  double *dev_x = nullptr;
  double *dev_y = nullptr; // result = A * X

  usparse::var_csr_desc<int, double> d_csr_desc_var =
      init_device_data(h_csr_desc, &x_double[0], &dev_x, &dev_y); // 申请内存
  usparse::csr_mat_descr<double, int> d_csr_desc = d_csr_desc_var;

  // warming up start
  gmres_float(d_csr_desc, &x_double[0], &b_double[0], dev_x, dev_y);
  USPARSE_HIP_API_CHECK(hipMemset(dev_x, 0, h_csr_desc.rows * sizeof(double))); // reset the solve result.
  // warming up end.

  auto res = gmres_float(d_csr_desc, &x_double[0], &b_double[0], dev_x, dev_y);
  // auto res = gmres(d_csr_desc, &x_double[0], &b_double[0], &dev_A, dev_x, dev_y); 
  // GMRES算法核心，需重点优化

  // destory data
  destory_device_data(d_csr_desc_var, dev_y, dev_x);

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
