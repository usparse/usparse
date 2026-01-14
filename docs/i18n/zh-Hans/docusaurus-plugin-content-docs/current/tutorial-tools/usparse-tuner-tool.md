---
sidebar_position: 3
---

# usparse Auto-tuner: CsrMV auto tuner

这是一个用于自动调优 CsrMV 性能的工具。
它可以针对特定的稀疏矩阵，遍历算法中的所有编译期的参数组合，并给出每一个参数组合对应的性能。
目前支持对 line-enhance CsrMV 算法进行性能调优，对应的可执行文件时 `usparse_tuner_csrmv_line_enhance`。

该工具通常随 `usparse` 项目一起构建，在CMake 的配置阶段，需要将 CMake 的选项 `USPARSE_AUTO_TUNER_ENABLE` 设置成 `ON`。CMake 构建完成后，可以在构建目录下的 `utuner/bin/` 找到对应的可执行文件（如：`usparse_tuner_csrmv_line_enhance`）。

## 命令行参数
这里，以 line-enhance 算法为例，说明性能调优工具的使用

命令行参数如下：
```bash
./usparse_tuner_csrmv_line_enhance -m <matrix_path> [-f <format>] [-o <output_format>]
```
我们，可通过 ./usparse_tuner_csrmv_line_enhance --help 更新命令行参数信息。
其中大部分参数和 benchmark 工具一致，具体如下：
|参数 | 长参数 | 描述  |
| --- | -- | --|
| `-m` | `--matrix` |矩阵文件路径 (必须指定) | 无 |
| `-f` | `--format` | 矩阵格式，支持 `mtx` (Matrix Market) 或 `bin2` (二进制) 。默认值是 `mtx`。|
| `-o` |`--output` |性能数据输出格式，支持 `stdout` 或 `csv`, 默认值是 `stdout`。 |
| `-h` |`--help` |显示帮助信息 |


## 运行示例（单矩阵）
单个矩阵运行示例，演示上述工具的使用，以遍历 line-enhance CsrMV 的所有参数组合并获取每一个参数组合对应的性能。
```bash
./build/utuner/bin/usparse_tuner_csrmv_line_enhance -m ./data/matrix.bin2 -f bin2 -o csv
```
这里，其使用 csv 格式输出性能测试报告，读取的矩阵是 .bin2 二进制格式。

## 运行示例（批量矩阵测试）
对于大规模的矩阵集，可以结合 suitesparse-dl 工具生成 Slurm 批处理脚本进行测试。

### 1. 生成 Slurm 批处理脚本

准备一个模板文件 `template.sh` （模板可基于 `tools/suitesparse-dl/template.example.sh` 文件进行修改），然后运行：
```bash
# 假设 .bin2 文件在 ./dl_bin2 目录下
suite-dl gen --data ./dl_bin2 --output tune_batch.sh --tpl template.example.sh --type bin2
```

其中，参考的 Slurm 模板 (template.sh)如下：
```bash
#!/bin/sh
#SBATCH -J spmv_tuner
#SBATCH -o log-%j.log
#SBATCH -e log-%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --exclusive
#SBATCH -p newlarge
#SBATCH -t 05:00:00
#SBATCH --export=ALL

echo "Running tuning on {{.Size}} matrices"

{{range .Metas}}
RAW_PATH="{{.AbsPath}}"
# 某些版本的 suitesparse-dl 工具有 bug，生成的路径包含重复的.bin2，如`tet.bin2.bin2`。
FIXED_PATH="${RAW_PATH%.bin2.bin2}.bin2"

./build/utuner/bin/usparse_tuner_csrmv_line_enhance -m "$FIXED_PATH" -f bin2 -o csv
{{end}}
```

### 2. 提交作业
生成的脚本可以直接通过 sbatch 提交：
```bash
sbatch tune_batch.sh
```
