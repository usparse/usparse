# ustb-sparse

ustb-sparse (short name: usparse), a sparse library for multiple backends: hip and others.

## Build for HIP

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

## Ecosystem
- [Performance-Explorer](https://github.com/genshen/Performance-Explorer)
- spmv-acc: the high performance and Open Source library for SpMV on GPUs. spmv-acc is a part of usparse.

# spmv-acc (old: https://github.com/hpcde/spmv-acc)

HIP acceleration for SpMV solver.

## Citing SpMV-acc
Please cite SpMV-acc in your publications if it helps your research:
```bib
# For github user: you can also click the link in this repository landing page at the right sidebar, with the label "Cite this repository."
@inproceedings{chu_spmv_gpu:_2023,
 title = {Efficient Algorithm Design of Optimizing SpMV on GPU},
 isbn = {979-8-4007-0155-9/23/06},
 url = {http://doi.org/10.1145/3588195.3593002},
 doi = {10.1145/3588195.3593002},
 language = {en},
 urldate = {2023-6-20},
 booktitle = {Proceedings of the 32nd International Symposium on High-Performance Parallel and Distributed Computing (HPDC '23), June 16--23, 2023, Orlando, FL, USA},
 publisher = {ACM Press},
 author = {Chu, Genshen and He, Yuanjie and Dong, Lingyu and Ding, Zhezhao and Chen, Dandan and Bai, He and Wang, Xuesong and Hu, Changjun},
 year = {2023},
 numpages = {14},
 series = {HPDC '23},
 address = {Orlando, Florida},
 location = {Orlando, FL, USA},
 pages = {1--14},
}
```

## Build
### Pre-requirements
- [ROCM](https://rocmdocs.amd.com): version 3.x or higher. For example: `module load compiler/rocm/3.9.1`
- [HIP](https://github.com/ROCm-Developer-Tools/HIP)
- [CMake](https://cmake.org): version 3.6 or higher.

### Download dependency
Before building, we need to download [clipp](https://github.com/muellan/clipp) for command line arguments processing.
```bash
# pkg: https://github.com/genshen/pkg
pkg fetch
pkg install
```

### Build steps
- Build and verify on GPU side. 
(Note: make sure lib `rocsparse` is loaded and its version must be greater/equal than "1.19.4 for ROCm 4.1.0"):
```bash
CC=clang CXX=hipcc cmake -DDEVICE_SIDE_VERIFY_FLAG=ON -DCMAKE_BUILD_TYPE=Release -B./build-hip -S./
cmake --build ./build-hip
./build-hip/bin/spmv-cli examples/data/rajat03.csr -f csr
```

- Build and verify on CPU side:
```bash
CC=clang CXX=hipcc cmake -DCMAKE_BUILD_TYPE=Release -B./build-hip -S./
cmake --build ./build-hip
./build-hip/bin/spmv-cli examples/data/rajat03.csr -f csr
```

- Build by specifying a kernel strategy (e.g., use strategy `Adaptive`):
```bash
CC=clang CXX=hipcc cmake -DKERNEL_STRATEGY=ADAPTIVE -DCMAKE_BUILD_TYPE=Release -B./build-hip-adaptive -S./
cmake --build ./build-hip-adaptive
./build-hip-adaptive/bin/spmv-cli examples/data/rajat03.csr -f csr
```

### Build with benchmark
```bash
# please remember to change WARP_SIZE in hola-hip on rocm platform
cmake --list-presets 
cmake --preset=rocm-hipcc-benchmark # or use --preset=cuda-hipcc-benchmark
cmake --build --preset=rocm-hipcc-benchmark -j 4  # or use --preset=cuda-hipcc-benchmark
```

## For Developers
### Add a new kernel strategy
A **kernel strategy** is an algorithm for calculating SpMV on device side.  
You can specific another kernel strategy (algorithm) by following rules:
1. Edit [src/configure.cmake](src/configure.cmake) to add a kernel strategy checking (e.g. add a strategy named `awesome_spmv`).
   ```diff
   +elseif (KERNEL_STRATEGY_LOWER MATCHES "awesome_spmv")
   +    set(KERNEL_STRATEGY_AWESOME_SPMV ON)
   else ()
       MESSAGE(FATAL_ERROR "unsupported kernel strategy ${KERNEL_STRATEGY}")
   endif ()
   ```
2. Edit [src/building_config.h.in](src/building_config.h.in) for generating C/C++ **Macro** defines of the corresponding strategy.
   ```diff
   #cmakedefine KERNEL_STRATEGY_DEFAULT
   +#cmakedefine KERNEL_STRATEGY_AWESOME_SPMV
   ```
3. Edit file `src/acc/CMakeLists.txt` and add the kernel strategy name, then CMake can find the source files of the kernel strategy.
   e.g.,
   ```diff
   # all enabled strategies
   set(ENABLED_STRATEGIES
       default
   +   awesome_spmv
   ```
4. Create a new directory named `hip-awesome-spmv` (replace '_' in strategy name to '-') under `src/acc` directory 
   and place your code for the new strategy to this directory.

5. Add file `source_list.cmake` to directory `src/acc/hip-awesome-spmv` to include the source files of the new strategy.
    Please refer to file `src/acc/hip/source_list.cmake` for more details.

6. Edit file `src/acc/strategy_picker.cpp` to call the entry function of the corresponding strategy.
   e.g.,
   ```diff
   void sparse_spmv(int trans, const double alpha, const double beta, int m, int n, const int *rowptr, const int *colindex,
                    const double *value, const double *x, double *y) {
   #ifdef KERNEL_STRATEGY_DEFAULT
   default_sparse_spmv(trans, alpha, beta, m, n, rowptr, colindex, value, x, y);
   #endif
   +#ifdef KERNEL_STRATEGY_AWESOME_SPMV
   +awesome_sparse_spmv(trans, alpha, beta, m, n, rowptr, colindex, value, x, y);
   +#endif
   ```

# SpGEMM (old: https://git.hpcer.dev/PRA/spgemm/sparse)
> SpGEMM for GPU

## Build
### Pre-requirements
- [ROCM](https://rocmdocs.amd.com) for AMD gpu: version 3.5 or higher. For example: `module load compiler/rocm/3.9.1`.
- [CUDA](https://developer.nvidia.com/cuda-zone) for NVIDIA gpu.
- [HIP](https://github.com/ROCm-Developer-Tools/HIP): version 3.5 or higher.
- [CMake](https://cmake.org): version 3.15 or higher.

### Build steps
- Build and verify on GPU side.
(Note: make sure lib `rocsparse` is loaded):
```
CC=clang CXX=hipcc cmake -DCMAKE_BUILD_TYPE=Release -B./${build_dir} -S./
cmake --build ./${build_dir} -j8
```

- Build and verify on CPU side.
```
CC=clang CXX=hipcc cmake -DCMAKE_BUILD_TYPE=Release -DDEVICE_SIDE_VERIFY_FLAG=OFF -B./${build_dir} -S./
cmake --build ./${build_dir} -j8
```

## Run
```
./${build_dir}/bin/spgemm-hip ${csr_matrix_file} ${sparsity_of_B} ${random_send}
```

## For Developers
### Add a new kernel strategy
A **kernel strategy** is an algorithm for calculating SpGEMM on device side.
You can specific another kernel strategy (algorithm) by following rules:
1. Edit [src/configure.cmake](src/configure.cmake) to add a kernel strategy checking (e.g. add a strategy named `awesome_spgemm`).
   ```diff
   +elseif (KERNEL_STRATEGY_LOWER MATCHES "awesome_spgemm")
   +    set(KERNEL_STRATEGY_AWESOME_SPGEMM ON)
   else ()
       MESSAGE(FATAL_ERROR "unsupported kernel strategy ${KERNEL_STRATEGY}")
   endif ()
   ```
2. Edit [src/building_config.h.in](src/building_config.h.in) for generating C/C++ **Macro** defines of the corresponding strategy.
   ```diff
   #cmakedefine KERNEL_STRATEGY_DEFAULT
   +#cmakedefine KERNEL_STRATEGY_AWESOME_SPGEMM
   ```
3. Edit [src/spgemm/hip/CMakeLists.txt](src/spgemm/hip/CMakeLists.txt) and add the kernel strategy name, then CMake can find the source files of the kernel strategy.
   e.g.,
   ```diff
   # all enabled strategies
   set(ENABLED_STRATEGIES
       default
   +   awesome_spgemm
   ```
4. Create a new directory named `hip-awesome-spgemm` (replace '_' in strategy name to '-') under `src/spgemm/hip` directory and place your code for the new strategy to this directory.

5. Add file `source_list.cmake` to directory `src/spgemm/hip/hip-awesome-spgemm` to include the source files of the new strategy. Please refer to file `src/acc/hip/source_list.cmake` for more details.

6. Edit [src/spgemm/hip/api/spgemm_hip_api.cc](src/spgemm/hip/api/spgemm_hip_api.cc) to call the entry function of the corresponding strategy.
   e.g.,
   ```diff
   template <typename T, typename I>
   void sparse_gemm_hip(const hip_sparse_operation transA, const hip_sparse_operation transB, 
                       const T alpha, const T beta, const size_t m, const size_t n, const size_t k, 
                       const size_t nnzA, const I *pOffsetA, const I *pColIndexA, const T *pValueA,
                       const size_t nnzB, const I *pOffsetB, const I *pColIndexB, const T *pValueB, 
                       size_t *pNnzC, I *pOffsetC, I **ppColIndexC, T **ppValueC) {
   #ifdef KERNEL_STRATEGY_DEFAULT
   default_spgemm<T, I>(alpha, m, n, k, nnzA, pOffsetA, pColIndexA, pValueA, nnzB, pOffsetB, pColIndexB, pValueB, pNnzC, pOffsetC, ppColIndexC ppValueC)
   #endif
   +#ifdef KERNEL_STRATEGY_AWESOME_SPGEMM
   +awesome_spgemm<T, I>(alpha, m, n, k, nnzA, pOffsetA, pColIndexA, pValueA, nnzB, pOffsetB, pColIndexB, pValueB, pNnzC, pOffsetC, ppColIndexC ppValueC)
   +#endif
   ```
