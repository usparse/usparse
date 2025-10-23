---
sidebar_position: 1
---

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

### Build
```bash
cmake --preset=rocm-hipcc
cmake --build --preset=rocm-hipcc

# or: build usparse with benchmark feature.
cmake --preset=rocm-hipcc-benchmark
cmake --build --preset=rocm-hipcc-benchmark
# Note: When pkg tool is used, you can change cmake option `USPARSE_BUILD_DEP_BY_PKG` to `ON`
# to enable HOLA benchmarking.
```
