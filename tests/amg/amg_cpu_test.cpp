#include "amg.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using namespace usparse::amg;

// Release 也必须检查，不能使用可能被 NDEBUG 禁用的 assert。
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Exception, typename Function>
void expect_exception(Function action) {
  bool caught = false;
  try { action(); } catch (const Exception&) { caught = true; }
  require(caught, "expected exception was not raised");
}

double norm2(const Vector& v) {
  double norm = 0.0;
  for (double value : v) norm = std::hypot(norm, value);
  return norm;
}

double max_error(const Vector& x, const Vector& expected) {
  require(x.size() == expected.size(), "solution dimension mismatch");
  double error = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    require(std::isfinite(x[i]), "non-finite solution");
    error = std::max(error, std::abs(x[i] - expected[i]));
  }
  return error;
}

// 一维二阶差分矩阵。完整保存上下两侧，边界采用齐次 Dirichlet 条件。
CsrMatrix poisson1d(int n) {
  CsrMatrix A;
  A.n = n;
  A.row_ptr.push_back(0);
  for (int i = 0; i < n; ++i) {
    if (i > 0) { A.col_idx.push_back(i - 1); A.values.push_back(-1.0); }
    A.col_idx.push_back(i);
    A.values.push_back(2.0);
    if (i + 1 < n) { A.col_idx.push_back(i + 1); A.values.push_back(-1.0); }
    A.row_ptr.push_back(static_cast<int>(A.values.size()));
  }
  return A;
}

// 二维五点差分矩阵，side*side 个未知量；求解器本身不会使用这些坐标。
CsrMatrix poisson2d(int side) {
  CsrMatrix A;
  A.n = side * side;
  A.row_ptr.push_back(0);
  for (int i = 0; i < A.n; ++i) {
    int row = i / side, column = i % side;
    if (row > 0) { A.col_idx.push_back(i - side); A.values.push_back(-1.0); }
    if (column > 0) { A.col_idx.push_back(i - 1); A.values.push_back(-1.0); }
    A.col_idx.push_back(i);
    A.values.push_back(4.0);
    if (column + 1 < side) { A.col_idx.push_back(i + 1); A.values.push_back(-1.0); }
    if (row + 1 < side) { A.col_idx.push_back(i + side); A.values.push_back(-1.0); }
    A.row_ptr.push_back(static_cast<int>(A.values.size()));
  }
  return A;
}

DenseMatrix dense(const CsrMatrix& A) {
  DenseMatrix M(A.n, Vector(A.n, 0.0));
  for (int i = 0; i < A.n; ++i)
    for (int p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p) M[i][A.col_idx[p]] += A.values[p];
  return M;
}

DenseMatrix dense(const CsrOperator& A) {
  DenseMatrix M(A.rows, Vector(A.cols, 0.0));
  for (int i = 0; i < A.rows; ++i)
    for (int p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p) M[i][A.col_idx[p]] += A.values[p];
  return M;
}

CsrMatrix from_dense(const DenseMatrix& M) {
  CsrMatrix A;
  A.n = static_cast<int>(M.size());
  A.row_ptr.push_back(0);
  for (int i = 0; i < A.n; ++i) {
    for (int j = 0; j < A.n; ++j) {
      if (M[i][j] != 0.0) { A.col_idx.push_back(j); A.values.push_back(M[i][j]); }
    }
    A.row_ptr.push_back(static_cast<int>(A.values.size()));
  }
  return A;
}

// 测试参考运算使用稠密循环；与算法内部的稀疏 Galerkin 实现分开。
Vector multiply(const DenseMatrix& M, const Vector& x) {
  Vector y(M.size(), 0.0);
  for (std::size_t i = 0; i < M.size(); ++i)
    for (std::size_t j = 0; j < x.size(); ++j) y[i] += M[i][j] * x[j];
  return y;
}

void check_result(const SolveResult& result, const Vector& x, const Vector& expected) {
  require(result.converged, "AMG exhausted the cycle limit");
  require(max_error(x, expected) < 1e-5, "known exact solution was not recovered");
  require(result.residual_history.size() == static_cast<std::size_t>(result.cycles) + 1,
          "residual history length is incorrect");
  require(result.residual_history.back() == result.residual_norm, "history final residual mismatch");
}

void run_problem(int size, bool two_dimensional, bool history) {
  CsrMatrix A = two_dimensional ? poisson2d(size) : poisson1d(size);
  Vector exact(A.n), b(A.n);
  // 非常量精确解，避免只验证插值算子恰好能精确表示的常量模式。
  for (int i = 0; i < A.n; ++i) exact[i] = 1.0 + 0.2 * std::sin(0.1 * (i + 1));
  // 独立按差分公式构造 b，而不是调用求解器内部的 SpMV。
  for (int i = 0; i < A.n; ++i) {
    b[i] = (two_dimensional ? 4.0 : 2.0) * exact[i];
    if (two_dimensional) {
      if (i / size > 0) b[i] -= exact[i - size];
      if (i % size > 0) b[i] -= exact[i - 1];
      if (i % size + 1 < size) b[i] -= exact[i + 1];
      if (i / size + 1 < size) b[i] -= exact[i + size];
    } else {
      if (i > 0) b[i] -= exact[i - 1];
      if (i + 1 < A.n) b[i] -= exact[i + 1];
    }
  }
  auto data = amg_setup(A);
  Vector x;
  auto result = amg_solve(data, b, x, 2000, 1e-10, 1e-12);
  check_result(result, x, exact);

  // 再用差分公式独立检查真残差，确保停止判断不是递归中的粗层残差。
  Vector r = b;
  for (int i = 0; i < A.n; ++i) {
    double ax = (two_dimensional ? 4.0 : 2.0) * x[i];
    if (two_dimensional) {
      if (i / size > 0) ax -= x[i - size];
      if (i % size > 0) ax -= x[i - 1];
      if (i % size + 1 < size) ax -= x[i + 1];
      if (i / size + 1 < size) ax -= x[i + size];
    } else {
      if (i > 0) ax -= x[i - 1];
      if (i + 1 < A.n) ax -= x[i + 1];
    }
    r[i] -= ax;
  }
  require(norm2(r) <= 1.0001 * std::max(1e-12, 1e-10 * norm2(b)), "true residual exceeds tolerance");

  std::cout << std::setprecision(12)
            << "backend: CPU (host), multilevel aggregation AMG\n"
            << "matrix: " << (two_dimensional ? "2D Poisson" : "1D Poisson") << "\n"
            << "levels: " << data.levels.size() << "\n";
  for (std::size_t l = 0; l < data.levels.size(); ++l) {
    const auto& level = data.levels[l];
    std::cout << "  level " << l << ": n=" << level.A.n << ", nnz=" << level.A.values.size();
    if (l + 1 < data.levels.size())
      std::cout << ", P=" << level.P.rows << 'x' << level.P.cols
                << ", R=" << level.R.rows << 'x' << level.R.cols;
    std::cout << '\n';
  }
  std::cout << "cycles: " << result.cycles << "\nresidual_norm: " << result.residual_norm
            << "\nmax_solution_error: " << max_error(x, exact) << '\n';
  if (history) {
    for (std::size_t i = 0; i < result.residual_history.size(); ++i)
      std::cout << "residual[" << i << "]: " << result.residual_history[i] << '\n';
  }
  if (A.n <= 16) {
    std::cout << "x:";
    for (double value : x) std::cout << ' ' << value;
    std::cout << '\n';
  }
  std::cout << "PASS\n";
}

// 对每一层独立验证 P0、R=P^T、连通聚合体及 A_next=R*A*P。
void check_hierarchy(const AMGHierarchy& data) {
  require(data.levels.size() >= 3, "test matrix did not construct a genuinely multilevel hierarchy");
  for (std::size_t l = 0; l + 1 < data.levels.size(); ++l) {
    const auto& level = data.levels[l];
    int n = level.A.n, c = data.levels[l + 1].A.n;
    require(c > 0 && c < n, "hierarchy did not strictly reduce the dimension");
    require(level.P.rows == n && level.P.cols == c && level.R.rows == c && level.R.cols == n,
            "transfer dimensions are inconsistent");
    require(level.aggregate.size() == static_cast<std::size_t>(n), "aggregate map size mismatch");
    auto A = dense(level.A), P0 = dense(level.P0), P = dense(level.P), R = dense(level.R);
    require(level.P0.rows == n && level.P0.cols == c, "tentative P0 dimensions are incorrect");
    std::vector<int> sizes(c, 0);
    for (int i = 0; i < n; ++i) {
      int group = level.aggregate[i];
      require(group >= 0 && group < c, "invalid aggregate index");
      ++sizes[group];
      double sum = 0.0;
      for (int j = 0; j < c; ++j) {
        require(P0[i][j] == (j == group ? 1.0 : 0.0), "tentative P0 does not match aggregates");
        require(R[j][i] == P[i][j], "R is not the transpose of P");
        sum += P0[i][j];
      }
      require(sum == 1.0, "P0 failed to reproduce a constant vector");
    }
    for (int group = 0; group < c; ++group) {
      require(sizes[group] > 0 && sizes[group] <= data.options.max_aggregate_size,
              "empty or oversized aggregate");
      // 在组内沿原矩阵的非零边走一遍，确保没有把不连通变量硬凑成一组。
      int seed = 0;
      while (level.aggregate[seed] != group) ++seed;
      std::vector<int> reached{seed};
      std::vector<bool> seen(n, false);
      seen[seed] = true;
      for (std::size_t head = 0; head < reached.size(); ++head) {
        int i = reached[head];
        for (int j = 0; j < n; ++j) {
          if (!seen[j] && level.aggregate[j] == group && A[i][j] != 0.0) {
            seen[j] = true;
            reached.push_back(j);
          }
        }
      }
      require(static_cast<int>(reached.size()) == sizes[group], "disconnected aggregate");
    }

    DenseMatrix AP(n, Vector(c, 0.0)), expected(c, Vector(c, 0.0));
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
        for (int k = 0; k < c; ++k) AP[i][k] += A[i][j] * P[j][k];
    for (int i = 0; i < c; ++i)
      for (int j = 0; j < n; ++j)
        for (int k = 0; k < c; ++k) expected[i][k] += R[i][j] * AP[j][k];
    DenseMatrix actual = dense(data.levels[l + 1].A);
    for (int i = 0; i < c; ++i)
      for (int j = 0; j < c; ++j)
        require(std::abs(actual[i][j] - expected[i][j]) <= 1e-11 * std::max(1.0, std::abs(expected[i][j])),
                "coarse matrix differs from independent dense R*A*P");
  }
  const auto& leaf = data.levels.back();
  require(leaf.aggregate.empty() && leaf.P0.values.empty() && leaf.P.values.empty() && leaf.R.values.empty(),
          "coarsest level must not have transfer operators");
}

void run_hierarchy_tests() {
  auto data = amg_setup(poisson1d(127));
  check_hierarchy(data);
  require(data.levels.back().A.n <= data.options.coarse_size, "1D coarsening stopped too early");
  AMGOptions options;
  options.strength_threshold = 0.0;
  check_hierarchy(amg_setup(poisson2d(12), options));
  // 非平滑分支也必须真正建立多层，且最终 P 应严格等于 P0。
  options.prolongation_sweeps = 0;
  auto unsmoothed = amg_setup(poisson1d(127), options);
  check_hierarchy(unsmoothed);
  for (const auto& level : unsmoothed.levels)
    require(level.P.values == level.P0.values && level.P.col_idx == level.P0.col_idx,
            "disabled interpolation smoothing changed P0");

  options.prolongation_sweeps = 2;

  options.max_levels = 2;
  auto limited = amg_setup(poisson1d(127), options);
  require(limited.levels.size() == 2, "max_levels was not respected");
  options.max_levels = 1;
  auto one = amg_setup(poisson1d(17), options);
  Vector exact(17, 1.0), b(17, 0.0), x;
  b.front() = b.back() = 1.0;
  auto result = amg_solve(one, b, x);
  require(one.levels.size() == 1 && result.cycles == 1, "single-level direct solve failed");
  check_result(result, x, exact);

  options = AMGOptions{};
  options.strength_threshold = 1.0; // Poisson 的相邻强度为 0.5，没有可聚合的强边。
  auto stalled = amg_setup(poisson1d(31), options);
  require(stalled.levels.size() == 1, "no-reduction hierarchy did not stop");
  options = AMGOptions{};
  options.coarse_size = 64;
  require(amg_setup(poisson1d(31), options).levels.size() == 1, "coarse_size was ignored");

  // 纯对角图无法聚合：应直接逐项求解，且不能创建 n*n 的稠密缓存。
  CsrMatrix diagonal;
  diagonal.n = 257;
  diagonal.row_ptr.push_back(0);
  for (int i = 0; i < diagonal.n; ++i) {
    diagonal.col_idx.push_back(i);
    diagonal.values.push_back(2.0 + i);
    diagonal.row_ptr.push_back(i + 1);
  }
  auto isolated = amg_setup(diagonal);
  require(isolated.coarse_is_diagonal && isolated.coarse_cholesky.empty(), "diagonal fallback failed");
  x.clear();
  check_result(amg_solve(isolated, diagonal.values, x), x, Vector(diagonal.n, 1.0));
  std::cout << "AMG hierarchy, transfer and Galerkin checks: PASS\n";
}

void run_regression() {
  // 在至少三层结构上检查停止条件和建立结果复用。
  auto data = amg_setup(poisson1d(63));
  Vector b(63, 0.0), exact(63, 1.0), x = exact;
  b.front() = b.back() = 1.0;
  auto result = amg_solve(data, b, x);
  require(result.converged && result.cycles == 0, "exact initial guess failed");
  Vector zero_rhs(63, 0.0);
  result = amg_solve(data, zero_rhs, x);
  check_result(result, x, Vector(63, 0.0));
  require(result.residual_norm <= 1e-12, "absolute tolerance failed for zero RHS");
  x.assign(63, 0.0);
  result = amg_solve(data, b, x, 0);
  require(!result.converged && result.cycles == 0 && max_error(x, Vector(63, 0.0)) == 0.0,
          "zero-budget solve changed the initial guess");
  const DenseMatrix factor_before = data.coarse_cholesky;
  const auto fine_before = data.levels.front().A.values;
  amg_vcycle(data, b, x);
  require(max_error(x, exact) < 1.0, "one recursive V-cycle did not improve the solution");
  b.front() = b.back() = 2.0;
  x.clear();
  check_result(amg_solve(data, b, x, 2000, 1e-10), x, Vector(63, 2.0));
  require(data.coarse_cholesky == factor_before && data.levels.front().A.values == fine_before,
          "solve modified the hierarchy or factorization");

  // 变量置换：求解器只能依靠稀疏图，而不能依赖连续编号或几何坐标。
  auto original = dense(poisson1d(31));
  DenseMatrix permuted(31, Vector(31));
  Vector expected(31), rhs_original(31, 0.0), rhs(31);
  rhs_original.front() = rhs_original.back() = 1.0;
  for (int i = 0; i < 31; ++i) {
    int old_i = (i * 7) % 31;
    expected[i] = 1.0;
    rhs[i] = rhs_original[old_i];
    for (int j = 0; j < 31; ++j) permuted[i][j] = original[old_i][(j * 7) % 31];
  }
  x.clear();
  check_result(amg_solve(amg_setup(from_dense(permuted)), rhs, x, 2000, 1e-10), x, expected);

  // 多个互不连接的 SPD 块也应被完整保留。
  DenseMatrix blocks{{2, -1, 0, 0, 0}, {-1, 2, 0, 0, 0}, {0, 0, 3, 1, 0},
                     {0, 0, 1, 2, 0}, {0, 0, 0, 0, 5}};
  AMGOptions small;
  small.coarse_size = 1;
  expected = {1, 2, -1, 0.5, 3};
  rhs = multiply(blocks, expected);
  x.clear();
  check_result(amg_solve(amg_setup(from_dense(blocks), small), rhs, x), x, expected);

  // 合并乱序重复列和显式零，检查一般 CSR 输入的预处理。
  CsrMatrix duplicated;
  auto input = poisson1d(31);
  duplicated.n = input.n;
  duplicated.row_ptr.push_back(0);
  for (int i = 0; i < input.n; ++i) {
    for (int p = input.row_ptr[i + 1] - 1; p >= input.row_ptr[i]; --p) {
      for (int copy = 0; copy < 2; ++copy) {
        duplicated.col_idx.push_back(input.col_idx[p]);
        duplicated.values.push_back(input.values[p] / 2.0);
      }
    }
    duplicated.col_idx.push_back(i);
    duplicated.values.push_back(0.0);
    duplicated.row_ptr.push_back(static_cast<int>(duplicated.values.size()));
  }
  require(dense(amg_setup(duplicated).levels.front().A) == dense(input), "CSR normalization failed");

  expect_exception<std::invalid_argument>([] { amg_setup(CsrMatrix{}); });
  CsrMatrix bad = input;
  bad.row_ptr[1] = 1000;
  expect_exception<std::invalid_argument>([&] { amg_setup(bad); });
  bad = input; bad.col_idx[0] = input.n;
  expect_exception<std::invalid_argument>([&] { amg_setup(bad); });
  bad = input; bad.values[0] = 0.0;
  expect_exception<std::invalid_argument>([&] { amg_setup(bad); });
  bad = input; bad.values[1] = -0.5;
  expect_exception<std::invalid_argument>([&] { amg_setup(bad); });
  bad = input; bad.values[0] = std::numeric_limits<double>::infinity();
  expect_exception<std::invalid_argument>([&] { amg_setup(bad); });
  small.max_levels = 0;
  expect_exception<std::invalid_argument>([&] { amg_setup(input, small); });
  small = AMGOptions{}; small.strength_threshold = -0.1;
  expect_exception<std::invalid_argument>([&] { amg_setup(input, small); });
  expect_exception<std::runtime_error>([] { amg_setup(from_dense({{1, 2}, {2, 1}})); });
  expect_exception<std::runtime_error>([] { amg_setup(from_dense({{1, -1}, {-1, 1}})); });
  Vector bad_rhs(63, 0.0);
  bad_rhs[0] = std::numeric_limits<double>::quiet_NaN();
  expect_exception<std::invalid_argument>([&] { amg_solve(data, bad_rhs, x); });
  expect_exception<std::invalid_argument>([&] { Vector wrong(2); amg_vcycle(data, b, wrong); });
  expect_exception<std::invalid_argument>([&] { Vector guess; amg_solve(data, b, guess, -1); });
  expect_exception<std::invalid_argument>([&] { AMGHierarchy empty; Vector guess; amg_solve(empty, {}, guess); });
  std::cout << "AMG multilevel regression checks: PASS\n";
}

int parse_size(const std::string& text, int maximum) {
  std::size_t parsed = 0;
  int n = std::stoi(text, &parsed);
  if (parsed != text.size() || n < 1 || n > maximum)
    throw std::invalid_argument("matrix size is invalid or exceeds the demo limit");
  return n;
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--self-test") { run_regression(); return 0; }
    if (argc == 2 && std::string(argv[1]) == "--hierarchy-test") { run_hierarchy_tests(); return 0; }
    if (argc == 2 && std::string(argv[1]) == "--help") {
      std::cout << "Usage: amg_cpu_test [size] [--history]\n"
                   "       amg_cpu_test --poisson2d side [--history]\n"
                   "       amg_cpu_test --self-test | --hierarchy-test\n"
                   "Default: 1D size 127. Demo limits: 1D 1..4096, 2D side 1..64.\n";
      return 0;
    }
    bool history = argc > 1 && std::string(argv[argc - 1]) == "--history";
    int count = history ? argc - 1 : argc;
    if (count == 1) run_problem(127, false, history);
    else if (count == 2) run_problem(parse_size(argv[1], 4096), false, history);
    else if (count == 3 && std::string(argv[1]) == "--poisson2d")
      run_problem(parse_size(argv[2], 64), true, history);
    else throw std::invalid_argument("use --help to see supported arguments");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "AMG test failed: " << error.what() << '\n';
    return 1;
  }
}
