# ustb-sparse

ustb-sparse (short name: usparse), a sparse library for multiple backends: hip and others.

## Build for HIP
```bash
# build spmv-acc
cd third-party/spmv-acc
pkg fetch
pkg install
cd ../../

# build usparse
cmake -B ./build -S ./ -DUSPARSE_ENABLE_HIP=ON -DCMAKE_BUILD_TYPE=Release
cmake --build ./build -j 4
cmake install ./build # install lib
```

## Ecosystem
- [Performance-Explorer](https://github.com/genshen/Performance-Explorer)
- spmv-acc: the high performance and Open Source library for SpMV on GPUs. spmv-acc is a part of usparse.
