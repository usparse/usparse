---
sidebar_position: 1
---

# Installation

## Build for HIP/ROCm
```bash
pkg fetch --features=rocm
pkg install

cmake --preset=rocm-hipcc
cmake --build --preset=rocm-hipcc

# or: build usparse with benchmark feature.
cmake --preset=rocm-hipcc-benchmark
cmake --build --preset=rocm-hipcc-benchmark
```

如果是在海光DCU的DTK环境下，请使用这两个 CMake Preset：`rocm-dcc-benchmark` 或者 `rocm-dcc`。  

例如，在DCU上构建支持benchmark特性的 usparse, 可以这样:
```bash
cmake --preset=rocm-dcc-benchmark
cmake --build --preset=rocm-dcc-benchmark
```
