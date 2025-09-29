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
