---
sidebar_position: 1
---

# Benchmark 工具

## 1. 如何使用Benchmark进行测试

当使用usparse时，如果需要让Benchmark起作用，需要在编译时选择到`benchmark`参数，如下：

```bash
# build usparse with benchmark feature.
cmake --preset=rocm-hipcc-benchmark
cmake --build --preset=rocm-hipcc-benchmark
```

对于各种计算类型，涉及到不同的参数输入，下面进行分类阐述；

### SpMV/SpMM

两种计算类型涉及的各种名称所代表的含义

CsrMV：读取矩阵为csr格式的SpMV算法，路径位于`usparse/benchmarks/csrmv`;

CooMV：读取矩阵为coo格式的SpMV算法，路径位于`usparse/benchmarks/coomv`；

CsrMM：读取矩阵为csr格式的SpMM算法，路径位于`usparse/benchmarks/csrmm`；

CooMM：读取矩阵为coo格式的SpMM算法，路径位于`usparse/benchmarks/coomm`；

生成的benchmark的可执行文件位于`usparse/build/bin`

该文件夹下可执行文件与算法的映射为：

CsrMV：csrmv-gpu-benchmark

CooMV：coomv-gpu-benchmark

CsrMM：csrmm-gpu-benchmark

CooMM：coomm-gpu-benchmark

该部分情况写为如下表格：

| 算法名称 | 矩阵格式 | 计算类型 | 源码路径               | 可执行文件名称 (位于 usparse/build/bin) |
| -------- | -------- | -------- | ---------------------- | --------------------------------------- |
| CsrMV    | CSR      | SpMV     | usparse/benchmarks/bin | csrmv-gpu-benchmark                     |
| CooMV    | COO      | SpMV     | usparse/benchmarks/bin | coomv-gpu-benchmark                     |
| CsrMM    | CSR      | SpMM     | usparse/benchmarks/bin | csrmm-gpu-benchmark                     |
| CooMM    | COO      | SpMM     | usparse/benchmarks/bin | coomm-gpu-benchmark                     |

#### 如何运行

以csrmv-gpu-benchmark为例，在使用benchmark时，通过在可执行文件后输入help参数可以得到具体可输入的参数说明，如下：

```bash
./csrmv-gpu-benchmark -h|--help
```

参数说明：

| 参数名称   | 短选项 | 长选项   | 是否带值 | 含义/用途                                                    |
| ---------- | ------ | -------- | -------- | ------------------------------------------------------------ |
| `help`     | `-h`   | `--help` | ❌        | 显示帮助信息后正常退出。                                     |
| `mtx_path` | `-m`   | `--mtx`  | ✅ 必填   | 指定待测试的矩阵文件路径（Matrix Market 或 bin2）。          |
| `fmt`      | `-f`   | `--fmt`  | ✅ 可选   | 指定矩阵文件格式：<br>• `mtx`（Matrix Market，默认）<br>• `bin2`（CSR 二进制 v2）。 |
| `fp`       | （无） | `--fp`   | ✅ 可选   | 指定浮点精度：<br>• `fp32`（单精度）<br>• `fp64`（双精度，默认）。 |

根据参数表，可以给出benchmark可执行文件的运行示例：

```bash
./csrmv-gpu-benchmark -m [matrix_A_filename] -f mtx --fp fp64
```

``./csrmv-gpu-benchmark``：矩阵A文件的存放路径，示例表示csrmv-gpu-benchmark位于当前文件夹下；

`-m matrix_A_filename`：指定读取的矩阵A的文件路径及文件名称；

``-f：mtx``：用于表示后续将要读取的矩阵A的文件类型为mtx格式；

  `--fp fp64`：指定的浮点精度为双精度。

### SpTRSM

CsrSM：读取矩阵为csr格式的SpTRSM算法，路径位于`usparse/benchmarks/csrsm`;

生成的benchmark的可执行文件位于`usparse/build/bin`，该文件夹下可执行文件与算法的映射为：

#### 如何运行

### SpTRSV

<!-- TODO:  -->

### SpGEMM

Csr_SpGEMM：读取矩阵为csr格式的SpGEMM算法，路径位于`usparse/benchmarks/csr_spgemm`;该文件夹下可执行文件与算法的映射为：
Csr_SpGEMM：`usparse/build/bin/spgemm-benchmark`。

#### 如何运行



## 2. Benchmark测试结果的参数说明

### SpMV

当执行完测试后，生成的测试结果共带有18个参数，下面对一些重要参数进行简单的介绍。

`matrix name`：测试用到的矩阵名称

`strategy name`：测试时的计算类型（SpMV或者其他计算）使用的策略名称

`rows`：矩阵的行数

`cols`：矩阵的列数

`nnz`：矩阵的非零元素数

`nnz/row`：平均每行的非零元素数

`GB/s(calc_time)`：实际达到的内存带宽，反映对内存子系统压力

`GFLOPS(calc_time)`：实际达到的浮点性能，反映计算单元利用率

`mid calc cost`：计算的平均时间开销

`mid total cost`：总的平均时间开销

`max_error`：与对比的CPU侧/GPU侧的基准代码的结算结果的最大绝对误差

### SpTRSM



### SpTRSV



### SpGEMM





<!-- TODO:  -->
