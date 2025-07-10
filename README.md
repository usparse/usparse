# SpGEMM
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
