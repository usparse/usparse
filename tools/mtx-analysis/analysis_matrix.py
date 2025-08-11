import os
import struct
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF
from PIL import Image
matplotlib.rcParams['font.family'] = ['SimHei']  # 黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

class SpM:
    def __init__(self):
        self.nrows = 0
        self.ncols = 0
        self.nnz = 0
        self.rows = None
        self.cols = None
        self.vals = None

def load_bin_to_csr(file_path):
    A = SpM()
    with open(file_path, "rb") as f:
        A.nrows = struct.unpack('I', f.read(4))[0]
        A.ncols = struct.unpack('I', f.read(4))[0]
        A.nnz = struct.unpack('I', f.read(4))[0]

        A.rows = np.frombuffer(f.read(4 * (A.nrows + 1)), dtype=np.uint32)
        A.cols = np.frombuffer(f.read(4 * A.nnz), dtype=np.uint32)
        A.vals = np.frombuffer(f.read(8 * A.nnz), dtype=np.float64)
    return A

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

def analyze_matrix(A, name_prefix="Matrix"):
    stats = {}
    stats["id"] = name_prefix
    stats["nrows"] = A.nrows
    stats["ncols"] = A.ncols
    stats["nnz"] = A.nnz
    stats["density"] = A.nnz / (A.nrows * A.ncols)
    stats["val_min"] = A.vals.min()
    stats["val_max"] = A.vals.max()
    stats["val_mean"] = A.vals.mean()
    stats["val_std"] = A.vals.std()

    row_nnz = np.diff(A.rows)
    stats["row_nnz_mean"] = row_nnz.mean()
    stats["row_nnz_std"] = row_nnz.std()
    stats["row_nnz_max"] = row_nnz.max()
    stats["row_nnz_min"] = row_nnz.min()


    return stats

def process_all_matrices(directory):
    bin_files = sorted([f for f in os.listdir(directory) if f.endswith(".bin")])
    print(f"Found {len(bin_files)} binary matrix files.")

    summary = []

    for i, fname in enumerate(bin_files):
        path = os.path.join(directory, fname)
        A = load_bin_to_csr(path)
        stats = analyze_matrix(A, name_prefix=f"Matrix_{i+1:02d}")
        summary.append(stats)

    df = pd.DataFrame(summary)
    df.to_csv("summary.csv", index=False)
    print("Summary saved to summary.csv")



# 替换为你的文件夹路径
process_all_matrices("D:\sparseMatrixSet")

def identify_anomalies(summary_csv="summary.csv", output="anomaly_report.csv"):
    import pandas as pd

    df = pd.read_csv(summary_csv)
    anomaly_rows = []

    for _, row in df.iterrows():
        reasons = []

        if row["row_nnz_std"] > row["row_nnz_mean"]:
            reasons.append("行非零分布极不均")

        if row["density"] < 1e-7:
            reasons.append("密度过低")
        elif row["density"] > 1e-2:
            reasons.append("密度过高")

        if row["row_nnz_max"] > 10 * max(1, row["row_nnz_min"]):
            reasons.append("最小最大行非零差异极大")

        block_stat_file = f"{row['id']}_block_rownnz_avgstd.png"

        if reasons:
            anomaly_rows.append({
                "Matrix ID": row["id"],
                "Shape": f"{row['nrows']}×{row['ncols']}",
                "NNZ": row["nnz"],
                "Density": f"{row['density']:.2e}",
                "RowNNZ Mean": f"{row['row_nnz_mean']:.2f}",
                "RowNNZ Std": f"{row['row_nnz_std']:.2f}",
                "RowNNZ Max/Min": f"'{int(row['row_nnz_max'])}/{int(row['row_nnz_min'])}'",
                "异常原因": "、".join(reasons)
            })

    if anomaly_rows:
        report_df = pd.DataFrame(anomaly_rows)
        report_df.to_csv(output, index=False)
        print(f"已识别 {len(anomaly_rows)} 个需要特殊处理的矩阵，报告已保存为 {output}")
    else:
        print("所有矩阵看起来结构正常，无需特殊处理。")

#  运行分析
identify_anomalies()
