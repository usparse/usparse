# SpTRSM算子测试说明文档

## 说明

- SpTRSM算子验证方式：CPU端验证，将计算出来的X矩阵在CPU端进行SpMM计算，得到B矩阵，与原始B矩阵进行对比，计算绝对误差和相对误差
- 误差阈值设置：单精度1E-5，双精度1E-12
- 性能验证：与ROCm6.0中的rocsparse（v3.0.2）对比，源码地址：https://github.com/ROCm/rocSPARSE/tree/rocm-6.0.0?tab=readme-ov-file

SpTRSM算子的测试程序可同时得到float和double结果。

## 编译和运行

### 准备环境

* dtk: version 22.04 or higher
* HIP：version 3.5 or higher
* CMake：version 3.15 or higher
* 依赖库：rocsparse

我们测试使用的 `乌镇机器`，加载的环境如下：

```shell
1) compiler/devtoolset/7.3.1   2) mpi/hpcx/2.11.0/gcc-7.3.1   
3) compiler/dtk/24.04          4) compiler/cmake/3.23.3
```

### 编译

测试程序位于源码主目录下，main.cpp为入口。

#### 修改链接库rocsparse的路径

程序默认链接的rocsparse库是加载的dtk中的rocsparse。若需要链接其他版本的rocsparse（如 rocsparse v3.0.2），则需要进入主目录下的 `CMakeList.txt`修改rocsparse库的路径，如下：

```bash
......
#target_link_libraries(sptrsm-csr PUBLIC rocsparse)
target_link_libraries(sptrsm-csr /work/home/chendd20/public/software/rocsparse/rocsparse3.0.2-dtk24.04/lib/librocsparse.so)
```

#### 构建和编译

```bash
cd sptrsm-csr

#export CC=clang 
#export CXX=hipcc 
cmake -B./build -S./
cd build
make
```

编译完成后，当前路径下可以看到 `build`文件夹，可执行文件位于 `build`路径下，`sptrsm-csr`即为测试程序的可执行文件。

### 运行

#### 直接运行

```bash
cd sptrsm-csr/build
./sptrsm <rounds> <number of rhs> <1 or 0> <path of mtx> mm
# 第1个参数为重复运行的轮次，执行多轮取平均时间作为最终的执行时间
# 第2个参数为右端向量数
# 第3个参数，"1"代表矩阵A是下三角矩阵，"0"代表矩阵A是上三角矩阵
例如： ./sptrsm-csr 3 3 1 #表示执行3轮、待求解阵X为3列，稀疏阵A为下三角阵，若A为非下三角阵，则处理成下三角阵后再求解AX=B。
# 第4个参数是输入的mtx文件的路径
# 第5个参数是输入文件的类型，mm表示输入的是.mtx文件，默认是.mtx文件
```

#### 使用脚本批量测试

```bash
cd sptrsm-csr
cd scripts
sbatch sptrsm_batch.sh
#注意修改脚本中的可执行文件路径，以及输入文件的路径
```

### 查看结果

> 查看生成的csv文件查看测试结果。

测试结果相关参数说明（仅展示与项目相关） :

> Matrix: 矩阵名称
>
> Rows(x)、Cols: 矩阵行数(单向量解的个数)、列数
>
> Time(us): 执行一次该算法的平均总花费时间，单位为us
>
> MaxAbs：GPU端与CPU端x的最大绝对误差值
>
> MaxRel: GPU端与CPU端x的最大相对误差值
>
> 1e-5: 相对误差大于1e-5的解的个数
>
> 1e-12: 相对误差大于1e-12的解的个数
>
> ps：UF \ UD \ RF \ RD :无同步单、双精度，Rocsparse单、双精度

执行的结果日志将输出到build目录下
生成的 csv 文件查看测试结果(脚本提交生成的 **.csv**保存在脚本文件路径下)。

## 2 程序代码结构及算法

*sptrsm-csr* 中共包含到测试源码、算法源码、测试用例和脚本文件，本程序主要实现无同步的算法实现。
该算法对应一个warp处理一行的思路，将左侧稀疏阵的每行非零元进行划分，一个线程处理一个或多个非零元的计算，最后在对一个warp进行归约，以合并除对角线元素外所有的非零局部和。最后对规约后的结果乘该行对角线元素的倒数以求得对应x。

```bash
. 程序代码结构
├── CMakeLists.txt
├── include
│   ├── cli
│   │	├── csr_binary_reader.hpp
│   │	├── matrix_market_reader.hpp
│   │	├── sort_omp.hpp
│   │	├── sparse_format.h
│   │	└── types.h
│   ├── common.h
│   ├── matrix_utils.h
│   ├── kernels.h
│   ├── verification.h
│   └── rocsparse_test.h
├── src
│    ├── matrix_utils.cpp
│    ├── verification.cpp
│    ├── updown_test.cpp
│    ├── rocsparse_test.cpp
│    └── kernels.cpp
├── scripts
│    ├── sptrsm-batch.sh
│    └── start.sh
└── main.cpp
```
