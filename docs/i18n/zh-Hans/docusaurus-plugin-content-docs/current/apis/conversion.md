---
sidebar_position: 6
---

# Data Type Conversion

## 1. COO to CSR

```c++
void usparse::conversion::coo2csr(usparse_int m, usparse_int nnz, const usparse_int *coo_row_ind, const usparse_int *csr_row_ptr);
```

将稀疏的 COO 矩阵转换为稀疏的 CSR 矩阵。

`coo2csr`将包含行索引的 COO 数组转换为指向每行开头的行偏移量的 CSR 数组。假设 COO 行索引数组已排序。

### 参数

* coo_row_ind – 包含稀疏 COO 矩阵的行索引的元素数组。`nnz`
* nnz – 稀疏 CSR 矩阵的非零元素数。
* m – 稀疏 CSR 矩阵的行数。
* csr_row_ptr – [out] 元素数组，指向稀疏 CSR 矩阵每行的开头。`m+1`

### 返回值

无返回值