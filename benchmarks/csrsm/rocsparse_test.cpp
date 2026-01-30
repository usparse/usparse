#include "rocsparse_test.h"
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <rocsparse/rocsparse.h>

#include "common/usparse_utils.hpp"

// template <typename T>
// T* test_rocsparse_matrix(const MatrixData<T>& matrix_in, const T* h_b, int k,
//                         int h_rhs, int* out_m, float* out_time)
// {
//     // 实现rocsparse测试逻辑...
//     //行数
//     rocsparse_int m=matrix_in.rows;
//     *out_m = m;
//    	rocsparse_int n= h_rhs;

//     const int* hcsr_row_ptr = matrix_in.row_ptr.data();
//     const int* hcsr_col_ind = matrix_in.col_index.data();
//     const T* hcsr_val  = reinterpret_cast<const T*>(matrix_in.values.data());

//     std::vector<T> hC(m * n, 0.0);

//     // Scalar alpha
//     T alpha = 1.0;
//     rocsparse_int nnz = matrix_in.nnz;

//     // Offload data to device
//     int* dcsr_row_ptr;
//     int* dcsr_col_ind;
//     T* dcsr_val;
//     T* dB;
//     T* dC;
//     hipMalloc((void**)&dcsr_row_ptr, sizeof(int) * (m + 1));
//     hipMalloc((void**)&dcsr_col_ind, sizeof(int) * nnz);
//     hipMalloc((void**)&dcsr_val, sizeof(T) * nnz);
//     hipMalloc((void**)&dB, sizeof(T) * m * n);
//     hipMalloc((void**)&dC, sizeof(T) * m * n);

//     hipMemcpy(dcsr_row_ptr, hcsr_row_ptr, sizeof(int) * (m + 1), hipMemcpyHostToDevice);
//     hipMemcpy(dcsr_col_ind, hcsr_col_ind, sizeof(int) * nnz, hipMemcpyHostToDevice);
//     hipMemcpy(dcsr_val, hcsr_val, sizeof(T) * nnz, hipMemcpyHostToDevice);
//     hipMemcpy(dB, h_b, sizeof(T) * m * n, hipMemcpyHostToDevice);

//     rocsparse_handle     handle;
//     rocsparse_spmat_descr matA;
//     rocsparse_dnmat_descr matB;
//     rocsparse_dnmat_descr matC;

//     rocsparse_indextype row_idx_type = rocsparse_indextype_i32;
//     rocsparse_indextype col_idx_type = rocsparse_indextype_i32;
//     rocsparse_datatype  data_type = std::is_same<T, float>::value ? rocsparse_datatype_f32_r :
//     rocsparse_datatype_f64_r; rocsparse_datatype  compute_type = std::is_same<T, float>::value ?
//     rocsparse_datatype_f32_r : rocsparse_datatype_f64_r; rocsparse_index_base idx_base = rocsparse_index_base_zero;
//     rocsparse_operation trans_A = rocsparse_operation_none;
//     rocsparse_operation trans_B = rocsparse_operation_none;

//     rocsparse_create_handle(&handle);

//     // Create sparse matrix A
//     rocsparse_create_csr_descr(&matA,
//                             m,
//                             m,
//                             nnz,
//                             dcsr_row_ptr,
//                             dcsr_col_ind,
//                             dcsr_val,
//                             row_idx_type,
//                             col_idx_type,
//                             idx_base,
//                             data_type);

//     // Create dense matrix B
//     rocsparse_create_dnmat_descr(&matB,
//                                 m,
//                                 n,
//                                 m,
//                                 dB,
//                                 data_type,
//                                 rocsparse_order_column);

//     // Create dense matrix C
//     rocsparse_create_dnmat_descr(&matC,
//                                 m,
//                                 n,
//                                 m,
//                                 dC,
//                                 data_type,
//                                 rocsparse_order_column);
//     // 预热
//     for(int i=0; i<5; i++){
//         // Call spsv to get buffer size
//         size_t buffer_size;
//         rocsparse_spsm(handle,
//                     trans_A,
//                     trans_B,
//                     &alpha,
//                     matA,
//                     matB,
//                     matC,
//                     compute_type,
//                     rocsparse_spsm_alg_default,
//                     rocsparse_spsm_stage_buffer_size,
//                     &buffer_size,
//                     nullptr);

//         void* temp_buffer;
//         hipMalloc((void**)&temp_buffer, buffer_size);

//         // Call spsv to perform analysis
//         rocsparse_spsm(handle,
//                     trans_A,
//                     trans_B,
//                     &alpha,
//                     matA,
//                     matB,
//                     matC,
//                     compute_type,
//                     rocsparse_spsm_alg_default,
//                     rocsparse_spsm_stage_preprocess,
//                     &buffer_size,
//                     temp_buffer);

//         // Call spsv to perform computation
//         rocsparse_spsm(handle,
//                     trans_A,
//                     trans_B,
//                     &alpha,
//                     matA,
//                     matB,
//                     matC,
//                     compute_type,
//                     rocsparse_spsm_alg_default,
//                     rocsparse_spsm_stage_compute,
//                     &buffer_size,
//                     temp_buffer);
//         hipFree(temp_buffer);

//     }
//     // my_timer tm;
//     // tm.start();
//     double time1=0.0;
//     for(int i=0; i<k; i++){
//         // 开始计时
//         my_timer tm;
//         tm.start();
//         // Call spsv to get buffer size
//         size_t buffer_size;
//         rocsparse_spsm(handle,
//                     trans_A,
//                     trans_B,
//                     &alpha,
//                     matA,
//                     matB,
//                     matC,
//                     compute_type,
//                     rocsparse_spsm_alg_default,
//                     rocsparse_spsm_stage_buffer_size,
//                     &buffer_size,
//                     nullptr);

//         void* temp_buffer;
//         hipMalloc((void**)&temp_buffer, buffer_size);

//         // Call spsv to perform analysis
//         rocsparse_spsm(handle,
//                     trans_A,
//                     trans_B,
//                     &alpha,
//                     matA,
//                     matB,
//                     matC,
//                     compute_type,
//                     rocsparse_spsm_alg_default,
//                     rocsparse_spsm_stage_preprocess,
//                     &buffer_size,
//                     temp_buffer);

//         // Call spsv to perform computation
//         rocsparse_spsm(handle,
//                     trans_A,
//                     trans_B,
//                     &alpha,
//                     matA,
//                     matB,
//                     matC,
//                     compute_type,
//                     rocsparse_spsm_alg_default,
//                     rocsparse_spsm_stage_compute,
//                     &buffer_size,
//                     temp_buffer);
//         hipFree(temp_buffer);
//         hipDeviceSynchronize();
//         tm.stop();//结束计时
//         time1+=tm.time_use;
//         if(i==0)
//             hipMemcpy(hC.data(), dC, sizeof(T) * m * n, hipMemcpyDeviceToHost);
//     }

//     // tm.stop();//结束计时
//     //std::cout << "GPU rocsparse computation time: " << tm.time_use/k *1.0 << " us" << std::endl;
//     *out_time = time1/k *1.0;

//     // Copy result back to host
//     // hipMemcpy(hC.data(), dC, sizeof(T) * m * n, hipMemcpyDeviceToHost);
//     // 拷贝结果到新数组
//     T* result = new T[m * n];
//     memcpy(result, hC.data(), m * n * sizeof(T));

//     rocsparse_destroy_spmat_descr(matA);
//     rocsparse_destroy_dnmat_descr(matB);
//     rocsparse_destroy_dnmat_descr(matC);
//     rocsparse_destroy_handle(handle);

//     // Clear device memory
//     hipFree(dcsr_row_ptr);
//     hipFree(dcsr_col_ind);
//     hipFree(dcsr_val);
//     hipFree(dC);
//     // hipFree(temp_buffer);

//     return result;

// }
template <typename T>
T *test_rocsparse_matrix(const MatrixData<T> &matrix_in, const T *h_b, int k, int h_rhs, int *out_m, float *out_time) {
  // Initialize parameters
  rocsparse_int m = matrix_in.rows;
  *out_m = m;
  rocsparse_int nrhs = h_rhs;
  rocsparse_int nnz = matrix_in.nnz;

  // Extract CSR data
  const rocsparse_int *hcsr_row_ptr = matrix_in.row_ptr.data();
  const rocsparse_int *hcsr_col_ind = matrix_in.col_index.data();
  const T *hcsr_val = matrix_in.values.data();

  // Prepare host output
  std::vector<T> hB(m * nrhs);
  T alpha = static_cast<T>(1.0);

  // Allocate device memory
  rocsparse_int *dcsr_row_ptr = nullptr;
  rocsparse_int *dcsr_col_ind = nullptr;
  T *dcsr_val = nullptr;
  T *dB = nullptr;
  void *temp_buffer = nullptr;

  USPARSE_HIP_API_CHECK(hipMalloc((void **)&dcsr_row_ptr, sizeof(rocsparse_int) * (m + 1)));
  USPARSE_HIP_API_CHECK(hipMalloc((void **)&dcsr_col_ind, sizeof(rocsparse_int) * nnz));
  USPARSE_HIP_API_CHECK(hipMalloc((void **)&dcsr_val, sizeof(T) * nnz));
  USPARSE_HIP_API_CHECK(hipMalloc((void **)&dB, sizeof(T) * m * nrhs));

  USPARSE_HIP_API_CHECK(hipMemcpy(dcsr_row_ptr, hcsr_row_ptr, sizeof(rocsparse_int) * (m + 1), hipMemcpyHostToDevice));
  USPARSE_HIP_API_CHECK(hipMemcpy(dcsr_col_ind, hcsr_col_ind, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
  USPARSE_HIP_API_CHECK(hipMemcpy(dcsr_val, hcsr_val, sizeof(T) * nnz, hipMemcpyHostToDevice));
  USPARSE_HIP_API_CHECK(hipMemcpy(dB, h_b, sizeof(T) * m * nrhs, hipMemcpyHostToDevice));

  // Create ROCm handles and descriptors
  rocsparse_handle handle;
  rocsparse_mat_descr descr;
  rocsparse_mat_info info;

  rocsparse_create_handle(&handle);
  rocsparse_create_mat_descr(&descr);
  rocsparse_set_mat_fill_mode(descr, rocsparse_fill_mode_lower);
  rocsparse_set_mat_diag_type(descr, rocsparse_diag_type_non_unit);
  rocsparse_create_mat_info(&info);

  if constexpr (std::is_same<T, double>::value) {
    // Query buffer size
    size_t buffer_size;
    rocsparse_dcsrsm_buffer_size(handle, rocsparse_operation_none, rocsparse_operation_none, m, nrhs, nnz, &alpha,
                                 descr, dcsr_val, dcsr_row_ptr, dcsr_col_ind, dB, m, info, rocsparse_solve_policy_auto,
                                 &buffer_size);

    USPARSE_HIP_API_CHECK(hipMalloc(&temp_buffer, buffer_size));

    // Analysis phase
    rocsparse_dcsrsm_analysis(handle, rocsparse_operation_none, rocsparse_operation_none, m, nrhs, nnz, &alpha, descr,
                              dcsr_val, dcsr_row_ptr, dcsr_col_ind, dB, m, info, rocsparse_analysis_policy_reuse,
                              rocsparse_solve_policy_auto, temp_buffer);
    // Warm-up runs
    for (int i = 0; i < 5; ++i) {
      rocsparse_dcsrsm_solve(handle, rocsparse_operation_none, rocsparse_operation_none, m, nrhs, nnz, &alpha, descr,
                             dcsr_val, dcsr_row_ptr, dcsr_col_ind, dB, m, info, rocsparse_solve_policy_auto,
                             temp_buffer);
    }

    // Timing runs
    double total_time = 0.0;
    for (int i = 0; i < k; ++i) {
      my_timer tm;
      tm.start();
      rocsparse_dcsrsm_solve(handle, rocsparse_operation_none, rocsparse_operation_none, m, nrhs, nnz, &alpha, descr,
                             dcsr_val, dcsr_row_ptr, dcsr_col_ind, dB, m, info, rocsparse_solve_policy_auto,
                             temp_buffer);

      USPARSE_HIP_API_CHECK(hipDeviceSynchronize());
      tm.stop();
      total_time += tm.time_use;

      if (i == 0) {
        USPARSE_HIP_API_CHECK(hipMemcpy(hB.data(), dB, sizeof(T) * m * nrhs, hipMemcpyDeviceToHost));
      }
    }
    *out_time = static_cast<float>(total_time / k);
    USPARSE_HIP_API_CHECK(hipFree(temp_buffer));
  } else {
    // Query buffer size
    size_t buffer_size;
    rocsparse_scsrsm_buffer_size(handle, rocsparse_operation_none, rocsparse_operation_none, m, nrhs, nnz, &alpha,
                                 descr, dcsr_val, dcsr_row_ptr, dcsr_col_ind, dB, m, info, rocsparse_solve_policy_auto,
                                 &buffer_size);

    USPARSE_HIP_API_CHECK(hipMalloc(&temp_buffer, buffer_size));

    // Analysis phase
    rocsparse_scsrsm_analysis(handle, rocsparse_operation_none, rocsparse_operation_none, m, nrhs, nnz, &alpha, descr,
                              dcsr_val, dcsr_row_ptr, dcsr_col_ind, dB, m, info, rocsparse_analysis_policy_reuse,
                              rocsparse_solve_policy_auto, temp_buffer);
    // Warm-up runs
    for (int i = 0; i < 5; ++i) {

      rocsparse_scsrsm_solve(handle, rocsparse_operation_none, rocsparse_operation_none, m, nrhs, nnz, &alpha, descr,
                             dcsr_val, dcsr_row_ptr, dcsr_col_ind, dB, m, info, rocsparse_solve_policy_auto,
                             temp_buffer);
    }

    // Timing runs
    double total_time = 0.0;
    for (int i = 0; i < k; ++i) {
      my_timer tm;
      tm.start();

      rocsparse_scsrsm_solve(handle, rocsparse_operation_none, rocsparse_operation_none, m, nrhs, nnz, &alpha, descr,
                             dcsr_val, dcsr_row_ptr, dcsr_col_ind, dB, m, info, rocsparse_solve_policy_auto,
                             temp_buffer);

      USPARSE_HIP_API_CHECK(hipFree(temp_buffer));

      USPARSE_HIP_API_CHECK(hipDeviceSynchronize());
      tm.stop();
      total_time += tm.time_use;

      if (i == 0) {
        USPARSE_HIP_API_CHECK(hipMemcpy(hB.data(), dB, sizeof(T) * m * nrhs, hipMemcpyDeviceToHost));
      }
    }
    *out_time = static_cast<float>(total_time / k);
    USPARSE_HIP_API_CHECK(hipFree(temp_buffer));
  }

  // Check for zero pivot
  rocsparse_int pivot;
  rocsparse_csrsm_zero_pivot(handle, info, &pivot);

  // Copy result back
  T *result = new T[m * nrhs];
  std::copy(hB.begin(), hB.end(), result);

  // Clean up
  rocsparse_destroy_mat_info(info);
  rocsparse_destroy_mat_descr(descr);
  rocsparse_destroy_handle(handle);
  USPARSE_HIP_API_CHECK(hipFree(dcsr_row_ptr));
  USPARSE_HIP_API_CHECK(hipFree(dcsr_col_ind));
  USPARSE_HIP_API_CHECK(hipFree(dcsr_val));
  USPARSE_HIP_API_CHECK(hipFree(dB));
  // USPARSE_HIP_API_CHECK(hipFree(temp_buffer));

  return result;
}

// 显式实例化
template float *test_rocsparse_matrix<float>(const MatrixData<float> &, const float *, int, int, int *, float *);
template double *test_rocsparse_matrix<double>(const MatrixData<double> &, const double *, int, int, int *, float *);