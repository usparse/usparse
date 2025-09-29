---
sidebar_position: 1
---

# DataTypes

## 1. 基础数据类型

| usaprse 中的数据类型 | 对应 C++ 中的数据类型 |
| -- | -- |
| `usaprse::usparse_int` |  `int` |
| `usaprse::usparse_uint` |  `uint` |
| `usaprse::usparse_fp32` |  `float` |
| `usaprse::usparse_fp64` |  `double` |


## 2. 稀疏矩阵存储格式

关于稀疏矩阵的各种表示，可以参考相关资料，这里仅介绍 usparse 库中的相关 API（例如：https://docs.nvidia.com/cuda/cusparse/#matrix-formats ）。

### Coordinate (COO)
COO 格式的表示为： `usparse::coo_desc<I, T>`，其中 I 是索引，T 是数据类型。
例如，索引为 usaprse::usparse_int ，矩阵中的数据的类型为 usaprse::usparse_fp64，则可以表示为：`usparse::coo_desc<usaprse::usparse_int, usparse::usparse_fp64>`。

`usparse::coo_desc<I, T>` 的构造函数如下：
```cpp
// m: 行数，n: 列数；nnz: 非零元素数
usparse::coo_desc(const I m, const I n, const I nnz, const I *coo_row_index, const I *coo_col_index, const T *coo_values);
```

:::info
我们假定 COO 中的数据已经是按行索引排序的。对列索引的是否排序无要求。
:::

### Compressed Sparse Row (CSR)

COO 格式的表示为： `usparse::csr_desc<I, T>`，其中 I 是索引，T 是数据类型。
例如，索引为 usaprse::usparse_int ，矩阵中的数据的类型为 usaprse::usparse_fp64，则可以表示为：`usparse::csr_desc<usaprse::usparse_int, usparse::usparse_fp64>`。

`usparse::csr_desc<I, T>` 的构造函数如下：
```cpp
// m: 行数，n: 列数；nnz: 非零元素数
usparse::csr_desc(const I m, const I n, const I nnz, const I *csr_row_offset, const I *csr_col_index, const T *csr_values);
```

:::info
CSR中对列索引是否排序无要求。
:::

### Ellpack (ELL)

TODO:

## 3. 矩阵描述符
- `usparse::mat::usparse_mat_info`

### API：初始化矩阵描述符
```cpp
usparse::usparse_status usparse::mat::usparse_create_mat_info(usparse::mat::usparse_mat_info *info);
```
示例：
```cpp
usparse::mat::usparse_mat_info info;
usparse::mat::usparse_create_mat_info(&info);
```

## 4. 枚举类型
- usparse::usparse_status
- usparse::usparse_operation
