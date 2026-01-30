//
// Created by luxu on 2023/02/14.
//

#ifndef SPGEMM_BENCHMARK_CREAT_RANDOM_MATRIX_HPP
#define SPGEMM_BENCHMARK_CREAT_RANDOM_MATRIX_HPP

template <typename I, typename T> void generate_sparse_matrix(T *&spmat, I rows, I cols, T sparsity) {
  for (I i = 0; i < rows; i++) {
    for (I j = 0; j < cols; j++) {
      I x = rand() % 1000000;
      if (x < 1000000.0 * sparsity) {
        spmat[i * cols + j] = x / 1000000.0 + 1.0;
      }
    }
  }
}

template <typename I, typename T>
void creat_device_matrix(I rows, I cols, T *spmat, I *&drow_ptr_B, I *&dcol_idx_B, T *&dval_B) {
  size_t nonzeros = 0;

  for (I i = 0; i < rows; i++) {
    for (I j = 0; j < cols; j++) {
      if (spmat[i * cols + j] != 0) {
        nonzeros++;
      }
    }
  }

  I *offset = new I[rows + 1];
  I *colindex = new I[nonzeros];
  T *value = new T[nonzeros];

  I k = 0;
  I l = 0;

  for (I i = 0; i < rows; i++) {
    for (I j = 0; j < cols; j++) {
      if (j == 0) {
        offset[l++] = k;
      }
      if (spmat[i * cols + j] != 0) {
        colindex[k] = j;
        value[k] = spmat[i * cols + j];
        k++;
      }
    }
  }
  offset[l] = nonzeros;

  USPARSE_HIP_API_CHECK(hipMalloc((void **)&drow_ptr_B, (rows + 1) * sizeof(I)))
  USPARSE_HIP_API_CHECK(hipMalloc((void **)&dcol_idx_B, nonzeros * sizeof(I)))
  USPARSE_HIP_API_CHECK(hipMalloc((void **)&dval_B, nonzeros * sizeof(T)))
  USPARSE_HIP_API_CHECK(hipMemcpy(drow_ptr_B, offset, (rows + 1) * sizeof(I), hipMemcpyHostToDevice))
  USPARSE_HIP_API_CHECK(hipMemcpy(dcol_idx_B, colindex, nonzeros * sizeof(I), hipMemcpyHostToDevice))
  USPARSE_HIP_API_CHECK(hipMemcpy(dval_B, value, nonzeros * sizeof(T), hipMemcpyHostToDevice))

  std::cout << "nnz of matrix_B=" << nonzeros << std::endl;
}

template <typename I, typename T>
void create_random_matrix(csr_mtx<I, T> &matrix, I *&drow_ptr_B, I *&dcol_idx_B, T *&dval_B, T sparsityB) {
  T *matB = NULL;
  matB = new T[matrix.rows * matrix.cols];

  memset(matB, 0, matrix.rows * matrix.cols * sizeof(T));

  generate_sparse_matrix<I, T>(matB, matrix.cols, matrix.rows, sparsityB);

  creat_device_matrix<I, T>(matrix.cols, matrix.rows, matB, drow_ptr_B, dcol_idx_B, dval_B);
}

#endif // SPGEMM_BENCHMARK_CREAT_RANDOM_MATRIX_HPP
