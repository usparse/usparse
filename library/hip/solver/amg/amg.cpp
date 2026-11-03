#include "amg.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include<map>

namespace usparse {
namespace amg {
namespace {

// 使用 int 作为 CSR 下标；在转换 vector 长度前检查，避免整数溢出。
int csr_offset(std::size_t size) {
    if (size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("AMG: CSR storage exceeds int index range");
    }
    return static_cast<int>(size);
}

// 将稀疏矩阵的一行数据追加到 CSR 格式的矩阵中。
void append_row(CsrMatrix& A, const std::map<int, double>& row_data) {
    for (const auto& entry : row_data) {
        if (!std::isfinite(entry.second)) {
            throw std::runtime_error("AMG: non-finite matrix entry during assembly");
        }

        if (entry.second != 0.0) {
            A.col_idx.push_back(entry.first);
            A.values.push_back(entry.second);
        }
    }
    A.row_ptr.push_back(csr_offset(A.col_idx.size()));
}

// 公共计算工具，按 csr 每行的非零元素，计算 y=Ax。
Vector csr_matvec(int rows, const std::vector<int>& row_ptr, const std::vector<int>&col_idx, 
    const Vector& values, const Vector& x) {
    Vector y(rows, 0.0);
    for (int i = 0; i <rows; ++i) {
        for (int p = row_ptr[i]; p < row_ptr[i+1]; ++p) {
            y[i] += values[p] * x[col_idx[p]];
        }
    }
    return y;
}

Vector matvec(const CsrMatrix& A, const Vector& x) {
    return csr_matvec(A.n, A.row_ptr, A.col_idx, A.values, x);
}

Vector matvec(const CsrOperator& A, const Vector& x) {
    return csr_matvec(A.rows, A.row_ptr, A.col_idx, A.values, x);
}

// 建立误差方程的右端，计算残差 r = b - Ax
Vector residual(const CsrMatrix& A, const Vector& x, const Vector& b) {
    Vector r = matvec(A, x);
    for (int i = 0; i < A.n; ++i) {
        r[i] = b[i] - r[i];
    }
    return r;
}

// 计算残差范数 ||v||_2=sqrt(sum_i v_i^2)，并拒绝 NaN/Inf 或溢出
double residual_norm2(const Vector& v) {
    double norm = 0.0;
    for (double value : v) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("AMG: numerical breakdown");
        }
        norm = std::hypot(norm, value);
    }
    if (!std::isfinite(norm)) {
        throw std::runtime_error("AMG: vector norm overflow");
    }
    return norm;
}

// -------------------------------------------------------------------------------
// S0: 进入多层建立前，先把原始输入整理成统一的 CSR 形式
// 检查数组和列号是否合法，按行排序、合并重复列，并删除显式零元素。
CsrMatrix canoical_matrix(const CsrMatrix& input) {
    if (input.n <= 0 || input.row_ptr.size() != static_cast<std::size_t>(input.n + 1) || 
    input.col_idx.size() != input.values.size()) {
        throw std::invalid_argument("AMG: invalid CSR matrix dimensions");
    }

    if (input.row_ptr.front() != 0 || input.row_ptr.back() != csr_offset(input.values.size())) {
        throw std::invalid_argument("AMG: invalid CSR row_ptr");
    }

    for (int i = 0; i < input.n; ++i) {
        if (input.row_ptr[i] > input.row_ptr[i + 1] || input.row_ptr[i] < 0 ) {
            throw std::invalid_argument("AMG: CSR row_ptr must be non-decreasing");
        }
    }

    CsrMatrix A;
    A.n = input.n;
    A.row_ptr.push_back(0);
    for (int i = 0; i < A.n; ++i) {
        std::map<int, double> row_data;
        for (int j = input.row_ptr[i]; j < input.row_ptr[i + 1]; ++j) {
            int col = input.col_idx[j];
            double val = input.values[j];
            if (col < 0 || col >= A.n) {
                throw std::invalid_argument("AMG: CSR col_idx out of bounds");
            }
            if (!std::isfinite(val)) {
                throw std::invalid_argument("AMG: CSR values must be finite");
            }
            row_data[col] += val; // 合并重复列

        }
        append_row(A, row_data);
    }
    return A;
}

// 检查对角正定矩阵。输入已排序、合并重复列、删除显式零元素的 CSR 矩阵，返回对角向量。
// 用于 Jacobi 平滑算法：A = D + L + U，其中 D 是对角矩阵，L 是严格下三角矩阵，U 是严格上三角矩阵。
Vector check_symemetry_extract_diagonal(const CsrMatrix& A) {
    Vector diagonal(A.n, 0.0);
    for (int i = 0; i < A.n; ++i) {
        for (int p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p) {
            if (A.col_idx[p] ==i) {
                diagonal[i] = A.values[p];
            }
        }
        if (!std::isfinite(diagonal[i]) || diagonal[i] <= 0.0) {
            throw std::invalid_argument("AMG: a positive diagonal is required in every row");
        }
    }

    for (int i = 0; i < A.n; ++i) {
        for (int p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p) {
            int j = A.col_idx[p];
            double value = A.values[p];
            if (j == i) continue;
            
            auto begin = A.col_idx.begin() + A.row_ptr[j];
            auto end = A.col_idx.begin() + A.row_ptr[j + 1];
            auto position = std::lower_bound(begin, end, i);
            double transposed = 0.0;
            if (position != end && *position == i) {
                transposed = A.values[static_cast<std::size_t>(position - A.col_idx.begin())];
            }
            // 稀疏 RAP 的加法顺序可能不同；用对角尺度容纳接近零元素的舍入差。
            double scale = std::max(std::max(std::abs(value), std::abs(transposed)),
                                    std::max(diagonal[i], diagonal[j]));
            if (std::abs(value - transposed) > 1e-12 * scale) {
                throw std::invalid_argument("AMG: full symmetric CSR storage is required");
            }
        }
    }
    return diagonal;
}

// 验证options
void validate_options(const AMGOptions& options) {
    if (options.max_levels <= 0) {
        throw std::invalid_argument("AMG: max_levels must be positive");
    }
    if (options.coarse_size <= 0) {
        throw std::invalid_argument("AMG: coarse_size must be positive");
    }
    if (options.max_aggregate_size <= 0) {
        throw std::invalid_argument("AMG: max_aggregate_size must be positive");
    }
    if (options.strength_threshold < 0.0) {
        throw std::invalid_argument("AMG: strength_threshold must be non-negative");
    }
    if (options.prolongation_sweeps < 0) {
        throw std::invalid_argument("AMG: prolongation_sweeps must be non-negative");
    }
    if (options.prolongation_damping <= 0.0 || options.prolongation_damping >= 1.0) {
        throw std::invalid_argument("AMG: prolongation_damping must be in (0,1)");
    }
    if (options.pre_sweeps < 0) {
        throw std::invalid_argument("AMG: pre_sweeps must be non-negative");
    }
    if (options.post_sweeps < 0) {
        throw std::invalid_argument("AMG: post_sweeps must be non-negative");
    }
}

// 验证vector，即对amg_vcycle和amg_solve的输入向量进行检查，确保其大小与矩阵A的维度一致。
void validate_vector(const Vector& v, int size) {
    if (v.size() != static_cast<std::size_t>(size)) {
        throw std::invalid_argument("AMG: vector size does not match matrix dimension");
    }
    for (double val : v) {
        if (!std::isfinite(val)) {
            throw std::invalid_argument("AMG: vector contains non-finite values");
        }
    }
}

// graph[i] 保存变量 i 的邻居，pair 的两个成员依次是“邻居编号、连接强度”。
using StrengthGraph = std::vector<std::vector<std::pair<int, double>>>;

// S1: 识别强连接关系
// s_ij=|a_ij|/sqrt(a_ii*a_jj)，保留 s_ij>=threshold 的边，并按强度排序。
StrengthGraph build_strength_graph(const CsrMatrix& A, const Vector& diagonal, double threshold) {
    StrengthGraph graph(A.n);
    for (int i = 0; i < A.n; ++i) {
        for (int p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p) {
            int j = A.col_idx[p];
            if (i ==j) continue;
            double strength = std::abs(A.values[p]) / (std::sqrt(diagonal[i] * diagonal[j]));
            if (strength >= threshold) {
                graph[i].emplace_back(j, strength);
            }
        }
        // 先按照连接强度从大到小排序；如果连接强度相等，则按照节点编号从小到大排序。
        std::sort(graph[i].begin(), graph[i].end(), [](const std::pair<int, double>& a, 
                                                    const std::pair<int, double>& b) {
            if (a.second != b.second) {
                return a.second > b.second; // 从大到小排序
            }
            return a.first < b.first; // 节点编号从小到大排序
        });
    }
    return graph;
}

// S2: 在强连接图上进行聚合，返回每个变量的聚合体编号
std::vector<int> build_aggregates(const StrengthGraph& graph, int max_size, int& count) {
    std::vector<int> aggregates(graph.size(), -1);
    count = 0;

    for (int seed = 0; seed < static_cast<int>(aggregates.size()); ++seed) {
        if (aggregates[seed] != -1) continue;
        aggregates[seed] = count;
        std::vector<int> members{seed};
        for (std::size_t head = 0; head < members.size() && static_cast<int>(members.size()) < max_size; ++head) {
            for (const auto& neighbour : graph[members[head]]) {
                if (aggregates[neighbour.first] != -1) continue;
                members.push_back(neighbour.first);
                aggregates[neighbour.first] = count;
                if (static_cast<int>(members.size()) == max_size) break;
            }
        }
        ++count;
    }
    return aggregates;
}

// S3: 构建初始插值矩阵 P0
CsrOperator build_initial_prolongation(const std::vector<int>& aggregates, int coarse_size) {
    CsrOperator P;
    P.rows = csr_offset(aggregates.size());
    P.cols = coarse_size;
    P.row_ptr.reserve(aggregates.size() + 1);
    P.row_ptr.push_back(0);
    for (int i = 0; i < static_cast<int>(aggregates.size()); ++i) {
        int agg_id = aggregates[i];
        if (agg_id < 0 || agg_id >= coarse_size) {
            throw std::invalid_argument("AMG: aggregate id out of bounds");
        }
        P.col_idx.push_back(agg_id);
        P.values.push_back(1.0); // 初始插值矩阵的非零值为 1
        P.row_ptr.push_back(csr_offset(P.col_idx.size()));
    }
    return P;
}

// S4: 平滑初始插值 P0，形成供后续 S5/S6 使用的最终 P
CsrOperator smooth_prolongation(const CsrMatrix& A, const Vector& diagonal, CsrOperator P, const AMGOptions& options) {
    if (options.prolongation_sweeps == 0) return P; // 不需要平滑，直接返回初始插值矩阵。
    
    double bound = 0.0;
    for (int i = 0; i < A.n; ++i) {
        double sum = 0.0;
        for (int p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p) {
            sum += std::abs(A.values[p] /diagonal[i]);
            bound = std::max(bound, sum);
        }
    }
    double omega = options.prolongation_damping / bound;
    // 平滑循环
    for (int sweep = 0; sweep < options.prolongation_sweeps; ++sweep) {
        CsrOperator P_next;
        P_next.rows = P.rows;
        P_next.cols = P.cols;
        P_next.row_ptr.push_back(0);
        for (int i = 0; i < A.n; ++i) {
            std::map<int, double> row;
            // 先复制 P 的当前行，对应公式中的 I*P。
            for (int p = P.row_ptr[i]; p < P.row_ptr[i + 1]; ++p) {
                row[P.col_idx[p]] += P.values[p];
            }
            for (int ap = A.row_ptr[i]; ap < A.row_ptr[i + 1]; ++ap) {
                int j = A.col_idx[ap];
                // 再累加 -omega*(A_ij/A_ii)*P 的第 j 行，对应 -omega*D^-1*A*P。
                double factor = -omega * (A.values[ap] / diagonal[i]);
                for (int pp = P.row_ptr[j]; pp < P.row_ptr[j + 1]; ++pp) {
                                row[P.col_idx[pp]] += factor * P.values[pp];
                }
            }
            for (const auto& entry : row) {
                if (!std::isfinite(entry.second)) throw std::runtime_error("AMG: interpolation overflow");
                if (entry.second != 0.0) {
                    P_next.col_idx.push_back(entry.first);
                    P_next.values.push_back(entry.second);
                }
            }
            P_next.row_ptr.push_back(csr_offset(P_next.values.size()));
        }
        P = std::move(P_next);
    }
    return P;
}

// S5: 构造限制算子 R=P^T
CsrOperator transpose(const CsrOperator& P) {
    CsrOperator R;
    R.rows = P.cols;
    R.cols = P.rows;
    R.row_ptr.assign(static_cast<std::size_t>(R.rows + 1), 0);
    R.col_idx.resize(P.values.size());
    R.values.resize(P.values.size());
    
    // P 的第 column 列成为 R 的第 column 行；先把该行的元素数量放在后一格。
    for (int column : P.col_idx) {
        ++R.row_ptr[column + 1];
    }
    for (int i = 0; i < R.rows; ++i) {
        R.row_ptr[i + 1] += R.row_ptr[i];
    }

    // 前缀和后 row_ptr 给出行起点；next 保存填充过程中每行的下一个空位。
    std::vector<int> R_row_ptr_next = R.row_ptr; // 各行的填充位置跨 P 的行累计。
    for (int i = 0; i < P.rows; ++i) {
        for (int p = P.row_ptr[i]; p < P.row_ptr[i + 1]; ++p) {
            int output = R_row_ptr_next[P.col_idx[p]];
            R_row_ptr_next[P.col_idx[p]] = R_row_ptr_next[P.col_idx[p]] + 1;
            R.col_idx[output] = i;
            R.values[output] = P.values[p];
        }
    }
    return R;
}

// S6: 生成下一层需要求解的稀疏方阵 A_next=R*A*P。
CsrMatrix galerkin_product(const CsrOperator& R, const CsrMatrix& A, const CsrOperator& P) {
  CsrMatrix coarse;
  coarse.n = R.rows;
  coarse.row_ptr.push_back(0);
  for (int c = 0; c < R.rows; ++c) {
    std::map<int, double> row;
    for (int rp = R.row_ptr[c]; rp < R.row_ptr[c + 1]; ++rp) {
      int i = R.col_idx[rp]; // R[c][i]：当前粗层行关联的细层行。
      for (int ap = A.row_ptr[i]; ap < A.row_ptr[i + 1]; ++ap) {
        int j = A.col_idx[ap]; // A[i][j]：细层矩阵连接到变量 j。
        for (int pp = P.row_ptr[j]; pp < P.row_ptr[j + 1]; ++pp) {
          // P[j][k] 再把变量 j 对应到粗层列 k。
          row[P.col_idx[pp]] += R.values[rp] * A.values[ap] * P.values[pp];
        }
      }
    }
    append_row(coarse, row);
  }
  return coarse;
}

// S8: 停止建立层级后，为最粗层准备直接求解器.
// 返回一个稠密矩阵，存储 Cholesky 分解得到的下三角矩阵 L。（乔列斯基分解）
// 对于实矩阵，标准 Cholesky 分解要求 \(A\) 是对称正定矩阵（SPD）。
DenseMatrix cholesky(const CsrMatrix& A) {
    DenseMatrix L(A.n, Vector(A.n, 0.0));
    for (int i = 0; i <A.n; ++i) {
        for (int p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p) {
            L[i][A.col_idx[p]] = A.values[p];
        }
    }

    for (int i = 0; i < A.n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double value = L[i][j];
            for (int k = 0; k < j; ++k) value -= L[i][k] * L[j][k];
                if (!std::isfinite(value) || (i == j && value <= 0.0)) {
                    throw std::runtime_error("AMG: coarsest matrix is not numerically positive definite");
                }
                L[i][j] = i == j ? std::sqrt(value) : value / L[j][j];
        }
        // 清除未使用的上三角，让缓存明确表示一个下三角矩阵。
        for (int j = i + 1; j < A.n; ++j) L[i][j] = 0.0;
    }
    return L;
}


// -------------------------------------------------------------------------------------------------
// V0: 递归到最粗层时，直接求解该层的 A_coarse*x=b。
Vector solve_coarse(const AMGHierarchy& data, const Vector& b) {
    const AMGLevel& coarse = data.levels.back();
    Vector x = b;
    if (data.coarse_is_diagonal) {
        for (int i = 0; i < coarse.A.n; ++i) x[i] /= coarse.diagonal[i];
        return x;
    }

    const DenseMatrix& L = data.coarse_cholesky;
    // 前向替换 L*y=b
    for (int i = 0; i < coarse.A.n; ++i) {
        for (int j = 0; j < i; ++j) {
            x[i] -= x[j] * L[i][j];
        }
        x[i] /= L[i][i];
    }
    // 后向替换 L^T*x=y
    for (int i = coarse.A.n - 1; i >= 0; --i) {
        for (int j = i + 1; j < coarse.A.n; ++j) {
            x[i] -= x[j] * L[j][i];
        }
        x[i] /= L[i][i];
    }
    return x;
}

// V1 前平滑、V7 后平滑：Gauss-Seidel 迭代当前层的解向量。
void smooth(const AMGLevel& level, const Vector& b, Vector& x, int sweeps, bool forward) {
    // 第一层循环：控制平滑次数
    for (int sweep = 0; sweep < sweeps; ++sweep) {
        // 第二层循环：遍历当前层所有变量
        for (int step = 0; step < level.A.n; ++step) {
            int i = forward ? step : (level.A.n - 1 - step);
            double rhs = b[i]; // 方程右端项
            // 第三层循环：遍历当前行的非零元素
            for (int p = level.A.row_ptr[i]; p < level.A.row_ptr[i + 1]; ++p) {
                int j = level.A.col_idx[p];
                if (j != i) rhs -= level.A.values[p] * x[j];
            }
            x[i] = rhs / level.diagonal[i];
        }
    }
}

// vcycle
void vcycle(const AMGHierarchy& data, std::size_t index, const Vector& b, Vector& x) {
    // V0:判断是否到达最粗层
    if (index + 1 == data.levels.size()) {
        x = solve_coarse(data, b);
        return;
    }
    // V1: 前平滑
    const AMGLevel& level = data.levels[index];
    smooth(level, b, x, data.options.pre_sweeps, true);
    // V2: 计算残差 r = b - Ax
    Vector r = residual(level.A, x, b);
    // V3: 限制残差 r 到下一层粗网格 r_c = R * r_f
    Vector coarse_rhs = matvec(level.R, r);
    // V4: 创建粗网格误差，并递归调用
    Vector coarse_error(level.P.cols, 0.0);
    vcycle(data, index + 1, coarse_rhs, coarse_error);
    // V5: 将粗网格误差插值回当前层
    Vector correction = matvec(level.P, coarse_error);
    // V6: 利用误差校正当前解
    for (int i = 0; i < level.A.n; ++i) {
        x[i] += correction[i];
    }
    // V7: 后平滑
    smooth(level, b, x,data.options.post_sweeps, false);
}

int fine_size(const AMGHierarchy& data) {
    if (data.levels.empty() || data.levels.front().A.n <= 0) {
        throw std::invalid_argument("AMG: hierarchy must be created by amg_setup");
    }
    return data.levels.front().A.n;
}

} // namespace

AMGHierarchy amg_setup(const CsrMatrix& input, const AMGOptions& options) {
    // S0：检查参数与初始化
    validate_options(options);
    AMGHierarchy data;
    data.options = options;
    CsrMatrix current = canoical_matrix(input);

    while (true) {
        AMGLevel level;
        level.A = std::move(current);
        level.diagonal = check_symemetry_extract_diagonal(level.A);

        // S7: 第一个停止条件
        bool stop =level.A.n <= options.coarse_size || 
                data.levels.size() + 1 >= static_cast<std::size_t>(options.max_levels);
        if (stop) {
            data.levels.push_back(std::move(level));
            break;
        }

        // S1：构建强连接图
        StrengthGraph graph = build_strength_graph(level.A, level.diagonal, options.strength_threshold);

        // S2：构建聚合组
        int count = 0;
        std::vector<int> aggregate = build_aggregates(graph, options.max_aggregate_size, count);
        
        // S7:  第二个停止条件：无法继续粗化
        if (count == level.A.n) {
            data.levels.push_back(std::move(level));
            break;
        }
        level.aggregate = std::move(aggregate);
        
        // S3：构造初始插值矩阵 P0
        level.P0 = build_initial_prolongation(level.aggregate, count);

        // S4：平滑插值矩阵
        level.P = smooth_prolongation(level.A, level.diagonal, level.P0, options);

        // S5：构造限制矩阵 R
        level.R = transpose(level.P);

        // S6：构造下一层粗网格矩阵
        current = galerkin_product(level.R, level.A, level.P);
        data.levels.push_back(std::move(level));  
    }

    // S8: 层级已完整，准备底层直接求解；V-cycle 只读取缓存，不重复分解。
    const CsrMatrix& coarse = data.levels.back().A;
    data.coarse_is_diagonal = coarse.values.size() == static_cast<std::size_t>(coarse.n);
    if (!data.coarse_is_diagonal) data.coarse_cholesky = cholesky(coarse);
    return data;
}

void amg_vcycle(const AMGHierarchy& data, const Vector& b, Vector& x) {
    int n = fine_size(data);
    validate_vector(b, n);
    validate_vector(x, n);
    vcycle(data, 0, b, x);
    residual_norm2(x);
}


// max_cycles，最多执行多少次 V-cycle；rtol，相对残差容差（Relative Tolerance）；atol，绝对残差容差（Absolute Tolerance）
SolveResult amg_solve(const AMGHierarchy& data, const Vector& b, Vector& x,
                        int max_cycles, double rtol, double atol) {
    // I0：检查输入。setup 负责建立矩阵层级，这里负责本次求解的 b、x 和容差。
    int n = fine_size(data);
    if (max_cycles < 0 || !std::isfinite(rtol) || !std::isfinite(atol) || rtol < 0.0 || atol < 0.0) {
        throw std::invalid_argument("AMG: invalid solve options");
    } 
    validate_vector(b, n);
    if (x.empty()) x.assign(n, 0.0);
    validate_vector(x, n);
    // I0：计算收敛阈值 target，相对阈值随 ||b|| 缩放；绝对阈值可处理 b=0 等情况，避免需要除以 ||b||。
    double target = std::max(atol, rtol * residual_norm2(b));
    if (!std::isfinite(target)) throw std::invalid_argument("AMG: tolerance overflow");

    SolveResult result;
    // I1：计算一次初始残差
    result.residual_norm = residual_norm2(residual(data.levels.front().A, x, b));
    result.residual_history.push_back(result.residual_norm);

    // I2：while 循环决定是否继续求解
    while (result.cycles < max_cycles && result.residual_norm > target) {
        vcycle(data, 0, b, x);
        ++result.cycles;
        // I3：每次 V-cycle 后重新计算残差
        result.residual_norm = residual_norm2(residual(data.levels.front().A, x, b));
        result.residual_history.push_back(result.residual_norm);
    }

    // I4：判断最终是否收敛
    result.converged = result.residual_norm <= target;
    return result;
}

} // namespace amg
} // namespace usparse
