---
sidebar_position: 1
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Installation

## Build for HIP/ROCm

### Download dependencies

If pkg tool is not avaiable on you system, we can run this script to download packages.
```bash
./scripts/dl_deps.sh
```
If pkg tool is used for manager dependency packages, you use following commands to download dependencies:
```bash
pkg fetch --features=rocm
pkg install
```

### Build without benchmark features
<Tabs groupId="current-backend" queryString>
  <TabItem value="dtk" label="DTK/HIP">
For DTK environment of Hygon DCU, use the CMake preset `rocm-dcc` (use dcc as compiler).  
e.g. to build on DCU without benchmark feature, run:
```bash
cmake --preset=rocm-dcc
cmake --build --preset=rocm-dcc
```
  </TabItem>
  <TabItem value="rocm" label="ROCm/HIP">
For ROCm environment, use the CMake preset `rocm-hipcc` (use hipcc as compiler).  
```bash
cmake --preset=rocm-dcc
cmake --build --preset=rocm-dcc
```
  </TabItem>
  <TabItem value="cuda" label="CUDA">
For CUDA environment, use the CMake preset `cuda-hipcc` (use nvcc as compiler).  
```bash
cmake --preset=cuda-hipcc
cmake --build --preset=cuda-hipcc
```
  </TabItem>
  <TabItem value="musa" label="MUSA">
For MUSA environment, use the CMake preset `musa-clang` (use clang/mcc as compiler).  
```bash
cmake --preset=cuda-hipcc
cmake --build --preset=cuda-hipcc
```
  </TabItem>
</Tabs>

### Build with benchmark features
To enable benchmark features, you can use the following commands to build usparse with benchmark.

<Tabs groupId="current-backend" queryString>
  <TabItem value="dtk" label="DTK/HIP">
For DTK environment of Hygon DCU, use the CMake preset `rocm-dcc-benchmark` (use dcc as compiler).  
e.g. to build on DCU with benchmark feature, run:
```bash
cmake --preset=rocm-dcc-benchmark
cmake --build --preset=rocm-dcc-benchmark
```
  </TabItem>
  <TabItem value="rocm" label="ROCm/HIP">
For ROCm environment, use the CMake preset `rocm-hipcc-benchmark` (use hipcc as compiler).  
```bash
cmake --preset=rocm-hipcc-benchmark
cmake --build --preset=rocm-hipcc-benchmark
```
  </TabItem>
  <TabItem value="cuda" label="CUDA">
For CUDA environment, use the CMake preset `cuda-nvcc-benchmark` (use nvcc as compiler).  
```bash
cmake --preset=cuda-nvcc-benchmark
cmake --build --preset=cuda-nvcc-benchmark
```
  </TabItem>
  <TabItem value="musa" label="MUSA">
For MUSA environment, use the CMake preset `musa-clang-benchmark` (use clang/mcc as compiler).  
```bash
cmake --preset=musa-clang-benchmark
cmake --build --preset=musa-clang-benchmark
```
  </TabItem>
</Tabs>

Note: When pkg tool is used, you can change cmake option `USPARSE_BUILD_DEP_BY_PKG` to `ON` to enable HOLA benchmarking.
