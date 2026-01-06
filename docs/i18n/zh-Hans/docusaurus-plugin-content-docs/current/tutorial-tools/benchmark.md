---
sidebar_position: 1
---

# Benchmark 工具

如果你编译 usparse 的时候，启用了 benchmark 特性，那么可以在构建目录下发现若干 *benchmark 可执行文件：
```bash
ls ./build/bin/*gpu-benchmark
./build/bin/coomm-gpu-benchmark
./build/bin/coomv-gpu-benchmark
./build/bin/csrmm-gpu-benchmark
./build/bin/csrmv-gpu-benchmark
./build/bin/gmres-gpu-benchmark
./build/bin/spgemm-benchmark
```
随后，就可以通过执行这些可执行文件，进行对应算子的不同算法的性能测试。

:::info
下面以 csrmv-gpu-benchmark 为例（其他benchmark 类似），演示 csrmv （csr 格式的SpMV）算子的 benchmark的使用。
:::

## benchmark 命令行选项

在可执行程序后续加上 `--help` 可以输出对应 benchmark 的命令行选项，例如：
```log
$ ./build/bin/csrmv-gpu-benchmark --help
  ./build/bin/csrmv-gpu-benchmark {OPTIONS}

    This is usparse benchmark.

  OPTIONS:

      -h, --help                        Display this help menu
      -m[matrix], --mtx=[matrix]        The matrix file
      -f[format], --fmt=[format]        The input matrix format. can be `mtx`
                                        (matrix market, default) or `bin2` (csr
                                        binary v2)
      --fp=[precision]                  The float percision used in benchmark.
                                        can be `fp32` or `fp64` (default)

    authors: genshen.
```
其中，各个选项的解释如下：
- `-m` `--mtx`，指定输入矩阵的路径，矩阵可以是 .mtx 格式（matrix market）或者 .bin2 格式。如 如：`-m ./test.mtx` `-fmtx`或 `--mtx ./test.mtx` 或 `-m../../sparse/s250r10-p6.mtx`。
- `-f` `--fmt`，指定矩阵的格式，值可以是 `mtx` 或者 `bin2`。 如：`-f mtx` `-fmtx`或 `--fmt mtx`。
- `--fp`，指定测试计算精度的算子API，目前支持单精度的测试和双精度的测试。 `--fp=fp32`指定测试单精度算子接口（如：csrmv 中的`usparse_scsrmv`接口）， `--fp=fp64` 指定测试双精度算子接口（如：csrmv中的`usparse_dcsrmv`接口）。 默认测试双精度接口。

## 输出日志
benchmark 的输出到标准输出中。
日志的第一行一般会打印测试的精度和矩阵路径，后面若干行是测试的性能信息（包括头部和一下若干行所表示的每一个算法的计算性能）。

我们可以对标准输出进行解析，以提取其中的性能信息（移除掉其他无关日志信息），将其转化为 .csv 格式。
例如：
```bash
$ cat log.log | grep PERFORMANCE # csrmv 为例
```
可能的结果如下：
```csv
PERFORMANCE,matrix name,strategy name,rows,cols,nnz,nnz/row,GB/s(calc_time),GFLOPS(calc_time),GB/s(total_time),GFLOPS(total_time),mid pre cost,mid calc cost,mid fixup(calc2) cost,mid destroy cost,mid total cost,first_failed_at,failed_count,max_error
PERFORMANCE,../../sparse/s250r10-p6.mtx,spmv-acc-adaptive,7525,277379,1207567,160.474,7.43473,1.97726,7.43473,1.97726,0,1221.45,0,0,1221.45,0,4158,0.0119019
PERFORMANCE,../../sparse/s250r10-p6.mtx,spmv-acc-flat,7525,277379,1207567,160.474,42.0114,11.1729,30.63,8.14603,80.32,216.16,0,0,296.48,0,4107,0.0119629
PERFORMANCE,../../sparse/s250r10-p6.mtx,spmv-acc-flat-v2,7525,277379,1207567,160.474,45.5882,12.1242,32.3036,8.59111,81.92,199.2,0,0,281.12,0,4107,0.0119629
PERFORMANCE,../../sparse/s250r10-p6.mtx,spmv-acc-light,7525,277379,1207567,160.474,19.4175,5.16407,19.4175,5.16407,0,467.68,0,0,467.68,0,4079,0.0119629
PERFORMANCE,../../sparse/s250r10-p6.mtx,spmv-acc-line,7525,277379,1207567,160.474,21.5316,5.72632,21.5316,5.72632,0,421.76,0,0,421.76,0,407,0.0119629
PERFORMANCE,../../sparse/s250r10-p6.mtx,spmv-acc-vector-row,7525,277379,1207567,160.474,21.0446,5.59679,21.0446,5.59679,0,431.521,0,0,431.521,0,4079,0.0119629
PERFORMANCE,../../sparse/s250r10-p6.mtx,spmv-acc-line-enhance,7525,277379,1207567,160.474,7.43577,1.97754,7.43577,1.97754,0,1221.28,0,0,1221.28,0,4158,0.0119019
PERFORMANCE,../../sparse/s250r10-p6.mtx,spmv-acc-adaptive-plus,7525,277379,1207567,160.474,152.574,40.5768,51.5507,13.7099,96.16,59.52,0,20.48,176.16,0,4075,0.0119019
PERFORMANCE,../../sparse/s250r10-p6.mtx,spmv-acc-flat-seg-sum,7525,277379,1207567,160.474,127.259,33.8444,66.9308,17.8002,64.32,71.36,0,0,135.68,0,3982,0.0118408
PERFORMANCE,../../sparse/s250r10-p6.mtx,rocSparse-vector-row,7525,277379,1207567,160.474,18.738,4.98336,13.1718,3.50304,173.12,484.64,0,31.68,689.44,0,3826,0.0119629
PERFORMANCE,../../sparse/s250r10-p6.mtx,rocSparse-adaptive,7525,277379,1207567,160.474,139.796,37.1788,17.2725,4.59361,419.2,64.96,0,41.6,525.76,0,4128,0.0119019
```
这是一个标准的 csv 格式，可以放到 MS Excel 等工具进行处理分析。

就csrmv 和 coomv 的性能输出而言（其他算子的含义可能会有差异），其中：
- mid pre cost： 表示前处理时间的中位数；
- mid calc cost： 表示算法计算时间的中位数；
- mid fixup(calc2) cost： 表示算法的额外计算时间的中位数（和上一条相加是整体计算时间）
- mid destroy cost： 表示算法的资源销毁（如释放内存、销毁 handle）部分时间的中位数
- mid total cost：前面前处理、计算、销毁的四个时间相加的中位数。
- first_failed_at：正确性验证相关。将 usparse 计算结果与基准结果（如CPU版本或者其他稀疏矩阵库）进行逐元素对比，第一个正确性对比不通过的元素出现的索引位置（SpMV中是结果向量中的正确性不符合的元素位置）。如果为 -1 则表示所有的元素验证均通过。
- failed_count：正确性验证相关。usparse 计算结果与基准相比，一共有多少个元素正确性验证不通过。
- max_error：正确性验证相关。usparse 计算结果与基准结果的逐元素对比中，绝对误差的最大值。
