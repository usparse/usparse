#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <benchmark/benchmark.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <rocsparse/rocsparse.h>

#include "../../../inc/sparseMatrix.hpp"
#include "../../unit/tests_utils.hpp"
#include "usparse_gmres.h"

std::string MATRIX_BASE_PATH = "/work/share/data/XDZS2025/GMRES/sparseMatrixSet/mtx";

void uspmv(int trans, const double alpha, const double beta, const csr_desc<int, double> h_csr_desc,
           const csr_desc<int, double> d_csr_desc, const double *dx, double *dy, const std::string &strategy) {
  if (strategy == "flat") {
    spmv_flat(trans, alpha, beta, h_csr_desc, d_csr_desc, dx, dy);
    hipDeviceSynchronize();
  } else if (strategy == "line") {
    spmv_line_enhance(trans, alpha, beta, d_csr_desc, dx, dy);
    hipDeviceSynchronize();
  } else if (strategy == "vec") {
    spmv_vector_row(trans, alpha, beta, d_csr_desc, dx, dy);
    hipDeviceSynchronize();
  } else {
    std::cerr << "Unknown strategy: " << strategy << ". Defaulting to vector_row implementation." << std::endl;
    spmv_vector_row(trans, alpha, beta, d_csr_desc, dx, dy);
    hipDeviceSynchronize();
  }
}

template <typename ValueType> void loadBinToCSR(SpM<ValueType> &A, const std::string &mtx_path) {
  std::ifstream binFile(mtx_path, std::ios::in | std::ios::binary);
  if (!binFile.is_open()) {
    std::cerr << "[ERROR] Cannot open file: " << mtx_path << std::endl;
    exit(EXIT_FAILURE);
  }
  binFile.read(reinterpret_cast<char *>(&A.nrows), sizeof(A.nrows));
  binFile.read(reinterpret_cast<char *>(&A.ncols), sizeof(A.ncols));
  binFile.read(reinterpret_cast<char *>(&A.nnz), sizeof(A.nnz));

  if (A.nrows <= 0 || A.ncols <= 0 || A.nnz <= 0) {
    std::cerr << "[ERROR] Invalid matrix dimensions" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (A.rows) {
    free(A.rows);
  }
  if (A.cols) {
    free(A.cols);
  }
  if (A.vals) {
    free(A.vals);
  }

  A.rows = (uint *)malloc((A.nrows + 1) * sizeof(uint));
  A.cols = (uint *)malloc(A.nnz * sizeof(uint));
  A.vals = (ValueType *)malloc(A.nnz * sizeof(ValueType));

  if (!A.rows || !A.cols || !A.vals) {
    std::cerr << "[ERROR] Host memory allocation failed" << std::endl;
    exit(EXIT_FAILURE);
  }

  binFile.read(reinterpret_cast<char *>(A.rows), (A.nrows + 1) * sizeof(uint));
  binFile.read(reinterpret_cast<char *>(A.cols), A.nnz * sizeof(uint));
  binFile.read(reinterpret_cast<char *>(A.vals), A.nnz * sizeof(ValueType));
  binFile.close();

  if (A.rows[A.nrows] != A.nnz) {
    std::cerr << "[ERROR] Invalid CSR format: last row pointer != nnz" << std::endl;
    exit(EXIT_FAILURE);
  }
}

static void BM_SpMV(benchmark::State &state) {
  const int matrix_num = state.range(0);
  const int strategy_idx = state.range(1);

  std::string strategy;
  switch (strategy_idx) {
  case 0:
    strategy = "flat";
    break;
  case 1:
    strategy = "line";
    break;
  case 2:
    strategy = "vec";
    break;
  default:
    strategy = "vec";
    break;
  }

  std::ostringstream oss;
  oss << MATRIX_BASE_PATH << matrix_num << ".bin";
  const std::string mtx_path = oss.str();

  SpM<double> h_A;
  loadBinToCSR(h_A, mtx_path);
  const int N = h_A.nrows;
  const int nnz = h_A.nnz;

  double *h_x = new double[N];
  double *h_y = new double[N];
  for (int i = 0; i < N; i++) {
    h_x[i] = sin(i);
  }
  memset(h_y, 0, N * sizeof(double));

  SpM<double> *dev_A;
  double *dev_x, *dev_y;

  TEST_HIP_CHECK(hipMalloc((void **)&dev_A, sizeof(SpM<double>)));
  TEST_HIP_CHECK(hipMalloc((void **)&dev_A->rows, (N + 1) * sizeof(uint)));
  TEST_HIP_CHECK(hipMalloc((void **)&dev_A->cols, nnz * sizeof(uint)));
  TEST_HIP_CHECK(hipMalloc((void **)&dev_A->vals, nnz * sizeof(double)));
  TEST_HIP_CHECK(hipMalloc((void **)&dev_x, N * sizeof(double)));
  TEST_HIP_CHECK(hipMalloc((void **)&dev_y, N * sizeof(double)));

  SpM<double> h_dev_A;
  h_dev_A.nrows = N;
  h_dev_A.ncols = h_A.ncols;
  h_dev_A.nnz = nnz;
  h_dev_A.rows = dev_A->rows;
  h_dev_A.cols = dev_A->cols;
  h_dev_A.vals = dev_A->vals;
  h_dev_A.on_device = true;

  TEST_HIP_CHECK(hipMemcpy(dev_A, &h_dev_A, sizeof(SpM<double>), hipMemcpyHostToDevice));
  TEST_HIP_CHECK(hipMemcpy(dev_A->rows, h_A.rows, (N + 1) * sizeof(uint), hipMemcpyHostToDevice));
  TEST_HIP_CHECK(hipMemcpy(dev_A->cols, h_A.cols, nnz * sizeof(uint), hipMemcpyHostToDevice));
  TEST_HIP_CHECK(hipMemcpy(dev_A->vals, h_A.vals, nnz * sizeof(double), hipMemcpyHostToDevice));
  TEST_HIP_CHECK(hipMemcpy(dev_x, h_x, N * sizeof(double), hipMemcpyHostToDevice));

  csr_desc<int, double> d_csr_desc(N, h_A.ncols, nnz, reinterpret_cast<const int *>(dev_A->rows),
                                   reinterpret_cast<const int *>(dev_A->cols), dev_A->vals);
  csr_desc<int, double> h_csr_desc(N, h_A.ncols, nnz, reinterpret_cast<const int *>(h_A.rows),
                                   reinterpret_cast<const int *>(h_A.cols), h_A.vals);

  for (auto _ : state) {
    uspmv(0, 1.0, 0.0, h_csr_desc, d_csr_desc, dev_x, dev_y, strategy);
  }

  if (dev_A) {
    if (dev_A->vals)
      TEST_HIP_CHECK(hipFree(dev_A->vals));
    if (dev_A->cols)
      TEST_HIP_CHECK(hipFree(dev_A->cols));
    if (dev_A->rows)
      TEST_HIP_CHECK(hipFree(dev_A->rows));
    TEST_HIP_CHECK(hipFree(dev_A));
    dev_A = nullptr;
  }
  if (dev_x)
    TEST_HIP_CHECK(hipFree(dev_x));
  if (dev_y)
    TEST_HIP_CHECK(hipFree(dev_y));

  delete[] h_x;
  delete[] h_y;
  if (h_A.rows) {
    free(h_A.rows);
    h_A.rows = nullptr;
  }
  if (h_A.cols) {
    free(h_A.cols);
    h_A.cols = nullptr;
  }
  if (h_A.vals) {
    free(h_A.vals);
    h_A.vals = nullptr;
  }

  const double mem_size = (sizeof(int) * (N + 1 + nnz) + sizeof(double) * (nnz + N + N)) * state.iterations();
  state.counters["Bandwidth"] = benchmark::Counter(mem_size, benchmark::Counter::kIsRate, benchmark::Counter::kIs1024);
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <matrix_num> <strategy>" << std::endl;
    std::cerr << "Available strategies: flat, line, vec" << std::endl;
    return 1;
  }

  int matrix_num = std::stoi(argv[1]);
  std::string strategy_str = argv[2];
  int strategy_index = -1;

  if (strategy_str == "flat")
    strategy_index = 0;
  else if (strategy_str == "line")
    strategy_index = 1;
  else if (strategy_str == "vec")
    strategy_index = 2;
  else {
    std::cerr << "Invalid strategy. Use flat, line or vec" << std::endl;
    return 1;
  }

  benchmark::RegisterBenchmark("SpMV", &BM_SpMV)->Args({matrix_num, strategy_index})->UseRealTime();

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();

  return 0;
}
