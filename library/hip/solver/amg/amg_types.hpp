#ifndef AMG_TYPE
#define AMG_TYPE

#include <vector>

namespace usparse {
namespace amg {

using Vector = std::vector<double>;
using DenseMatrix = std::vector<std::vector<double>>;

// 已有通用 csr 结构，后续考虑接入，这里先定义一个简单的以便测试。
struct CsrMatrix {
    int n = 0;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    Vector values;
};

struct CsrOperator {
    int rows = 0;
    int cols = 0; 
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    Vector values;
};

// AMG 参数结构体，包含多重网格的层数、粗化策略、平滑器参数等。
struct AMGOptions {
    int max_levels = 20;                // 包括最细层和最粗层
    int coarse_size = 8;              // 当前阶数 <= 此值时停止粗化      
    int max_aggregate_size = 4;          // 一个聚合体最多包含的当前层变量数
    double strength_threshold = 0.0;      // 默认保留所有非零边，仍优先吸收强连接邻居
    int prolongation_sweeps = 1;        // Jacobi 插值平滑次数，设为 0 可得到非平滑聚合。
    double prolongation_damping = 0.8;  // (0,1)，再除以 D^-1*A 的谱半径上界
    int pre_sweeps = 2;                 // 向粗层下降前的前向 Gauss-Seidel 次数
    int post_sweeps = 2;                // 返回细层后的后向 Gauss-Seidel 次数
};

// AMG 层级结构体，包含每一层的矩阵、对角元素、聚合体编号、插值矩阵和限制矩阵等信息。
struct AMGLevel {
    CsrMatrix A;          // 当前层矩阵
    Vector diagonal;      // 当前层对角元素
    std::vector<int> aggregate; // 当前层聚合体编号
    CsrOperator P0;       // 初始分片常数插值，用于理解聚合结果及检查常量再现
    CsrOperator P;        // 当前层到下一层的插值矩阵
    CsrOperator R;        // 下一层到当前层的限制矩阵
};

// AMG 完整层级结构体，包含所有层级信息和最粗层的 Cholesky 分解结果。
struct AMGHierarchy {
    AMGOptions options;            // AMG 参数
    std::vector<AMGLevel> levels; // 层级结构
    DenseMatrix coarse_cholesky;  // 缓存最粗层的 L，满足 A_coarse=L*L^T。
    bool coarse_is_diagonal = false; // 最粗层是否为对角矩阵
};

struct SolveResult {
    int cycles = 0;     // 多重网格循环次数
    double residual_norm = 0.0; // 最终残差范数
    bool converged = false; // 是否收敛
    Vector residual_history; // 残差历史
};

}
}

#endif // AMG_TYPE