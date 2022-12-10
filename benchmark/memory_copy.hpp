//
// Created by luxu on 2022/12/08.
//

#ifndef SPGEMM_BENCHMARK_MEMORY_MANAGEMENT_HPP
#define SPGEMM_BENCHMARK_MEMORY_MANAGEMENT_HPP

#include "rocsparse_csr2csc.hpp"

template <typename I, typename T>
void create_deivce_data(csr_mtx<I, T> &matrix, I *&drow_ptr_A, I *&dcol_idx_A, T *&dval_A, bool opB, I *&drow_ptr_B, I *&dcol_idx_B, T *&dval_B, I *&drow_ptr_C){
    std::cout<<"Preparing Device Memory for matrixA ..."<<std::endl;
    HIP_CHECK(hipMalloc((void **)&drow_ptr_A, (matrix.rows + 1) * sizeof(I)))
    HIP_CHECK(hipMalloc((void **)&dcol_idx_A, matrix.nnz * sizeof(I)))
    HIP_CHECK(hipMalloc((void **)&dval_A, matrix.nnz * sizeof(T)))
    HIP_CHECK(hipMemcpy(drow_ptr_A, matrix.row_ptr, (matrix.rows + 1) * sizeof(I), hipMemcpyHostToDevice))
    HIP_CHECK(hipMemcpy(dcol_idx_A, matrix.col_index, matrix.nnz * sizeof(I), hipMemcpyHostToDevice))
    HIP_CHECK(hipMemcpy(dval_A, matrix.values, matrix.nnz * sizeof(T), hipMemcpyHostToDevice))
    
    std::cout<<"Preparing Device Memory for matrixB ..."<<std::endl;
    if(opB){
        std::cout<<"MatrixB is transposition of matrixA"<<std::endl;
        rosparse_csr_matirx_turn<I, T>(matrix, drow_ptr_B, dcol_idx_B, dval_B);
    } else{
        std::cout<<"MatrixB is equal to matrixA"<<std::endl;
        HIP_CHECK(hipMalloc((void **)&drow_ptr_B, (matrix.rows + 1) * sizeof(I)))
        HIP_CHECK(hipMalloc((void **)&dcol_idx_B, matrix.nnz * sizeof(I)))
        HIP_CHECK(hipMalloc((void **)&dval_B, matrix.nnz * sizeof(T)))
        HIP_CHECK(hipMemcpy(drow_ptr_B, matrix.row_ptr, (matrix.rows + 1) * sizeof(I), hipMemcpyHostToDevice))
        HIP_CHECK(hipMemcpy(dcol_idx_B, matrix.col_index, matrix.nnz * sizeof(I), hipMemcpyHostToDevice))
        HIP_CHECK(hipMemcpy(dval_B, matrix.values, matrix.nnz * sizeof(T), hipMemcpyHostToDevice))
    }

    std::cout<<"Preparing Device Memory for matrixC's row_ptr ..."<<std::endl;
    HIP_CHECK(hipMalloc((void **)&drow_ptr_C, (matrix.rows + 1) * sizeof(I)))
}

#endif // SPGEMM_BENCHMARK_MEMORY_MANAGEMENT_HPP
