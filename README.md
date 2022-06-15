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
CC=clang CXX=hipcc cmake -DCMAKE_BUILD_TYPE=Release -DDEVICE_SIDE_VERIFY_FLAG=ON -B./${build_dir} -hip -S./
cmake --build ./${build_dir} -j8
```

- Build and verify on CPU side.
```
CC=clang CXX=hipcc cmake -DCMAKE_BUILD_TYPE=Release -DDEVICE_SIDE_VERIFY_FLAG=OFF -B./${build_dir} -hip -S./
cmake --build ./${build_dir} -j8
```

## Run
```
./${build_dir}/bin/spgemm-hip ${csr_matrix_file} ${sparsity_of_B} ${random_sand}
```

## For Developers
### Add a new kernel strategy
todo:
