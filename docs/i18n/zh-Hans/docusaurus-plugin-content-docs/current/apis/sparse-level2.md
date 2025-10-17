---
sidebar_position: 4
---

# Sparse Level2 Functions

## 1. CsrMV
计算 y = alpha * A *x + beta *y。其中，A 为CSR 格式存储的稀疏矩阵，x、y 均为稠密向量，alpha、beta 均为标量。

### DateType: usparse::spmv::usparse_csrmv_algorithms
```cpp
namespace usparse {
  namespace spmv {
    enum class usparse_csrmv_algorithms { Flat, LineEnhance, CsrAdaptivePlus, CsrVector, Adaptive }
  }
}
```

`usparse::spmv::usparse_csrmv_algorithms` 是一个枚举类型，表示可用于 CsrMV 的算法，可有在 usparse_csrmv_analysis 和 usparse_csrmv API中进行指定（且需要一致）。

### usparse_csrmv_analysis
```cpp
usparse::usparse_status usparse::spmv::usparse_scsrmv_analysis(
                            const usparse::usparse_operation trans,
                            const usparse::csr_desc<usparse_int, usparse_fp32> dev_csr_mat,
                            usparse::mat::usparse_mat_info *info,
                            const usparse::spmv::usparse_csrmv_algorithms algo);

usparse::usparse_status usparse::spmv::usparse_dcsrmv_analysis(
                            const usparse::usparse_operation trans,
                            const usparse::csr_desc<usparse::usparse_int, usparse::usparse_fp64> dev_csr_mat,
                            usparse::mat::usparse_mat_info *info,
                            const usparse::spmv::usparse_csrmv_algorithms algo);
```

CsrMV之前的矩阵分析步骤，某些算法需要进行额外的矩阵分析和内存申请操作。其中：
- Flat：需要申请一定的 device 端内存，需要在 device 上执行矩阵的预处理。
- LineEnhance：无 device 端内存申请，无预处理操作。
- CsrAdaptivePlus：需要申请 host 端内存，且需要在 host 端执行预处理。
- CsrVector：无 device 端内存申请，无预处理操作。

### usparse_csrmv
```cpp
usparse::usparse_status usparse::spmv::usparse_scsrmv(
                            const usparse::usparse_operation trans,
                            const usparse::usparse_fp32 alpha, 
                            const usparse::usparse_fp32 beta,
                            const usparse::csr_desc<usparse::usparse_int, usparse::usparse_fp32> dev_csr_mat,
                            const usparse::usparse_fp32 *x,
                            usparse::usparse_fp32 *y,
                            usparse::mat::usparse_mat_info *info,
                            const usparse::spmv::usparse_csrmv_algorithms algo);

usparse::usparse_status usparse::spmv::usparse_dcsrmv(
                            const usparse::usparse_operation trans,
                            const usparse::usparse_fp64 alpha,
                            const usparse::usparse_fp64 beta,
                            const usparse::csr_desc<usparse::usparse_int, usparse::usparse_fp64> dev_csr_mat,
                            const usparse::usparse_fp64 *x,
                            usparse::usparse_fp64 *y,
                            usparse::mat::usparse_mat_info *info,
                            const usparse::spmv::usparse_csrmv_algorithms algo);
```

在 device 上执行 CsrMV 计算，其中 `info` 参数为 usparse_csrmv_analysis api 的结果。

:::note
如果算法指定的是 `usparse::spmv::usparse_csrmv_algorithms::Flat`，计算结果可能是不稳定的（即不保证两次运行能产生完全一致的结果）。
:::

### usparse_csrmv_clear
```cpp
usparse::usparse_status usparse::spmv::usparse_csrmv_clear(usparse::mat::usparse_mat_info *info);
```

执行清理操作，如释放 usparse_csrmv_analysis 步骤申请的内存等。

## 2. CooMV

计算 y = alpha * A *x + beta *y。其中，A 为 COO 格式存储的稀疏矩阵，x、y 均为稠密向量，alpha、beta 均为标量。

### DateType: usparse::spmv::usparse_coomv_algorithms

```c++
namespace usparse {
  namespace spmv {
    enum class usparse_coomv_algorithms { NaiveFlat, Flat, LineEnhance, AdaptivePlus, Vector, Adaptive };
  }
}
```

`usparse::spmv::usparse_coomv_algorithms` 是一个枚举类型，表示可用于 CooMV 的算法，可有在 usparse_coomv_analysis 和 usparse_coomv API中进行指定（且需要一致）。

### usparse_coomv_analysis

```c++
usparse::usparse_status usparse_scoomv_analysis(const usparse_operation trans,
                                           const coo_desc<usparse_int, usparse_fp32> host_coo_mat,
                                           const coo_desc<usparse_int, usparse_fp32> dev_coo_mat,
                                           mat::usparse_mat_info *info, const usparse_coomv_algorithms algo);

usparse::usparse_status usparse_dcoomv_analysis(const usparse_operation trans,
                                           const coo_desc<usparse_int, usparse_fp64> host_coo_mat,
                                           const coo_desc<usparse_int, usparse_fp64> dev_coo_mat,
                                           mat::usparse_mat_info *info, const usparse_coomv_algorithms algo);
```

CsrMV之前的矩阵分析步骤，某些算法需要进行额外的矩阵分析和内存申请操作。其中：

* NaiveFlat：无额外的device端内存需求，无预处理操作。
* Flat：需要申请一定的 device 端内存，需要在 device 上执行矩阵的预处理。
* LineEnhance：需要申请一定的 device 端内存，需要在 device 上执行矩阵的预处理。
* AdaptivePlus：需要申请 host 端内存，且需要在 host 端执行预处理。
* Vector：需要申请一定的 device 端内存，需要在 device 上执行矩阵的预处理。
* Adaptive：需要申请 host 端内存，且需要在 host 端执行预处理。

### usparse_coomv

```c++
usparse::usparse_status usparse_scoomv(const usparse_operation trans, const usparse_fp32 alpha, const usparse_fp32 beta,
                                  const coo_desc<usparse_int, usparse_fp32> host_coo_mat,
                                  const coo_desc<usparse_int, usparse_fp32> dev_coo_mat, const usparse_fp32 *x,
                                  usparse_fp32 *y, mat::usparse_mat_info *info, const usparse_coomv_algorithms algo);

usparse::usparse_status usparse_dcoomv(const usparse_operation trans, const usparse_fp64 alpha, const usparse_fp64 beta,
                                  const coo_desc<usparse_int, usparse_fp64> host_coo_mat,
                                  const coo_desc<usparse_int, usparse_fp64> dev_coo_mat, const usparse_fp64 *x,
                                  usparse_fp64 *y, mat::usparse_mat_info *info, const usparse_coomv_algorithms algo);
```

COO 矩阵必须按行索引排序。

在 device 上执行 CooMV 计算，其中 `info` 参数为 usparse_coomv_analysis api 的结果。

:::note
如果算法指定的是 `usparse::spmv::usparse_coomv_algorithms::Flat`，计算结果可能是不稳定的（即不保证两次运行能产生完全一致的结果）。
:::

### usparse_coomv_clear

```c++
usparse::usparse_status usparse::spmv::usparse_coomv_clear(usparse::mat::usparse_mat_info *info);
```

执行清理操作，如释放 usparse_coomv_analysis 步骤申请的内存等。

<!-- TODO: -->
