---
sidebar_position: 1
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 构建安装

## 下载依赖项

如果您的系统上没有 `pkg`（https://github.com/genshen/pkg）工具，可以运行此脚本下载软件包。
```bash
./scripts/dl_deps.sh
```

如果使用 `pkg` 工具管理依赖包，则使用以下命令下载依赖项：
```bash
pkg fetch --features=rocm
pkg install
```

### 不启用基准测试功能的构建
<Tabs groupId="current-backend" queryString>
  <TabItem value="dtk" label="DTK/HIP">
对于 Hygon DCU 的 DTK 环境，使用 CMake preset `rocm-dcc`（使用 dcc 作为编译器）。  
例如，在 DCU 上构建且不启用基准测试功能，运行：
```bash
cmake --preset=rocm-dcc
cmake --build --preset=rocm-dcc
```
  </TabItem>
  <TabItem value="rocm" label="ROCm/HIP">
对于 ROCm 环境，使用 CMake preset `rocm-hipcc`（使用 hipcc 作为编译器）。  
```bash
cmake --preset=rocm-hipcc
cmake --build --preset=rocm-hipcc
```
  </TabItem>
  <TabItem value="cuda" label="CUDA">
对于 CUDA 环境，使用 CMake preset `cuda-hipcc`（使用 nvcc 作为编译器）。  
```bash
cmake --preset=cuda-hipcc
cmake --build --preset=cuda-hipcc
```
  </TabItem>
  <TabItem value="musa" label="MUSA">
对于 MUSA 环境，使用 CMake preset `musa-clang`（使用 clang/mcc 作为编译器）。  
```bash
cmake --preset=cuda-hipcc
cmake --build --preset=cuda-hipcc
```
  </TabItem>
</Tabs>

### 启用基准测试功能的构建
要启用基准测试功能，您可以使用以下命令构建带基准测试的 usparse。

<Tabs groupId="current-backend" queryString>
  <TabItem value="dtk" label="DTK/HIP">
对于 Hygon DCU 的 DTK 环境，使用 CMake preset `rocm-dcc-benchmark`（使用 dcc 作为编译器）。  
例如，在 DCU 上构建并启用基准测试功能，运行：
```bash
cmake --preset=rocm-dcc-benchmark
cmake --build --preset=rocm-dcc-benchmark
```
  </TabItem>
  <TabItem value="rocm" label="ROCm/HIP">
对于 ROCm 环境，使用 CMake preset `rocm-hipcc-benchmark`（使用 hipcc 作为编译器）。  
```bash
cmake --preset=rocm-hipcc-benchmark
cmake --build --preset=rocm-hipcc-benchmark
```
  </TabItem>
  <TabItem value="cuda" label="CUDA">
对于 CUDA 环境，使用 CMake preset `cuda-nvcc-benchmark`（使用 nvcc 作为编译器）。  
```bash
cmake --preset=cuda-nvcc-benchmark
cmake --build --preset=cuda-nvcc-benchmark
```
  </TabItem>
  <TabItem value="musa" label="MUSA">
对于 MUSA 环境，使用 CMake preset `musa-clang-benchmark`（使用 clang/mcc 作为编译器）。  
```bash
cmake --preset=musa-clang-benchmark
cmake --build --preset=musa-clang-benchmark
```
  </TabItem>
</Tabs>

注意：当使用 `pkg` 工具时，您可以将 CMake option `USPARSE_BUILD_DEP_BY_PKG` 设为 `ON` 以启用 HOLA 基准测试。
