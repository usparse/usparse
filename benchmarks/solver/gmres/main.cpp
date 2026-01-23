#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
#include <cstdint>
#include <tuple>
#include <cmath>
#include <iomanip>

#include "usparse_matrix.h"
#include "gmres.hpp"
#include "init.hpp"
#include "sparseMatrix.hpp"
#include "solver/gmres/kernels/usparse_gmres.h"

using RESULT = std::tuple<int, float, double>;

void loadBinToCSR(SpM<double> &A, std::string mtx_path) {
  std::ifstream binFile(mtx_path, std::ios::in | std::ios::binary);
  if (!binFile.is_open()) {
    std::cerr << "Error: Cannot open file " << mtx_path << std::endl;
    exit(1);
  }

  // 1. 头部处理 (自动跳过 .bin2 头)
  uint32_t magic = 0;
  binFile.read(reinterpret_cast<char*>(&magic), sizeof(magic));
  if (magic == 0x20211015) {
      binFile.seekg(12, std::ios::beg);
  } else {
      binFile.seekg(0, std::ios::beg);
  }

  // 2. 读取维度
  binFile.read(reinterpret_cast<char *>(&A.nrows), sizeof(A.nrows));
  binFile.read(reinterpret_cast<char *>(&A.ncols), sizeof(A.ncols));
  binFile.read(reinterpret_cast<char *>(&A.nnz), sizeof(A.nnz));

  // 3. 申请内存 (A.vals 始终为 double)
  A.rows = (uint *)malloc((A.nrows + 1) * sizeof(uint));
  A.cols = (uint *)malloc(A.nnz * sizeof(uint));
  A.vals = (double *)malloc(A.nnz * sizeof(double));

  // 4. 读取结构
  binFile.read(reinterpret_cast<char *>(A.rows), (A.nrows + 1) * sizeof(uint));
  binFile.read(reinterpret_cast<char *>(A.cols), A.nnz * sizeof(uint));

  // 5. 【核心修复】自适应读取数值 (Float -> Double 转换)
  std::streampos current_pos = binFile.tellg();
  binFile.seekg(0, std::ios::end);
  std::streampos end_pos = binFile.tellg();
  binFile.seekg(current_pos, std::ios::beg);

  long long remaining_bytes = end_pos - current_pos;
  long long needed_double = (long long)A.nnz * 8;
  long long needed_float  = (long long)A.nnz * 4;

  if (std::abs(remaining_bytes - needed_double) < 16) {
      // 刚好是 Double，直接读
      binFile.read(reinterpret_cast<char *>(A.vals), A.nnz * sizeof(double));
  } 
  else if (std::abs(remaining_bytes - needed_float) < 16) {
      // 是 Float，读入临时 buffer 并转换
      // std::cout << "Info: Converting Float to Double..." << std::endl;
      std::vector<float> temp_vals(A.nnz);
      binFile.read(reinterpret_cast<char *>(temp_vals.data()), A.nnz * sizeof(float));
      for(uint i=0; i<A.nnz; ++i) A.vals[i] = static_cast<double>(temp_vals[i]);
  } 
  else {
      // 大小不对，认为是 Pattern 矩阵，填 1.0
      // std::cout << "Info: Pattern Matrix detected, filling 1.0" << std::endl;
      for(uint i=0; i<A.nnz; ++i) A.vals[i] = 1.0;
  }

  // 6. 最终清洗 (修复原始数据中的 NaN)
  for(uint i=0; i<A.nnz; ++i) {
      if (std::isnan(A.vals[i]) || std::isinf(A.vals[i])) A.vals[i] = 0.0;
  }
  binFile.close();
}

int main(int argc, char *argv[]) {
  if (argc < 2) return 1;
  std::string mtx_name = argv[1];
  size_t last_slash = mtx_name.rfind('/');
  if (last_slash != std::string::npos) mtx_name = mtx_name.substr(last_slash + 1);

  SpM<double> A_double;
  loadBinToCSR(A_double, argv[1]);

  std::cout << mtx_name << ": M = " << A_double.nrows << ", N = " << A_double.ncols << std::endl;

  std::vector<double> x_double(A_double.nrows, 0);
  std::vector<double> b_double(A_double.nrows, 1);
  initialize(&A_double, &x_double[0], &b_double[0]);

  initDCU(0);

  SpM<double> dev_A;
  double *dev_x = nullptr, *dev_y = nullptr;
  var_csr_desc<int, double> d_var_csr_desc;
  init_device_data(&A_double, &x_double[0], &dev_A, &dev_x, &dev_y, &d_var_csr_desc);
  auto d_csr_desc = d_var_csr_desc.as_const();

  // Warmup
  gmres_float(&A_double, &x_double[0], &b_double[0], &dev_A, dev_x, dev_y, d_csr_desc);
  HIP_CHECK(hipMemset(dev_x, 0, A_double.nrows * sizeof(double)));
  
  // Real Run
  auto res = gmres_float(&A_double, &x_double[0], &b_double[0], &dev_A, dev_x, dev_y, d_csr_desc);

  destory_device_data(&dev_A, dev_y, dev_x);

  std::cout << "iters = " << std::get<0>(res) << ", time = " << std::get<1>(res) << "ms, resid = " << std::get<2>(res) << std::endl;

  // 写入文件以便记录
  std::ofstream outfile("gmres_time.txt", std::ios::app);
  if (outfile.is_open()) {
    outfile << mtx_name << " " << std::get<0>(res) << " " << std::get<1>(res) << " " << std::get<2>(res) << "\n";
    outfile.close();
  }
  return 0;
}