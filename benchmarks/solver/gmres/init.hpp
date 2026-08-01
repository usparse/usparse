#ifndef INITAL_H
#define INITAL_H

#include <cassert>
#include <cmath>
#include <iostream>
#include <string.h>
#include <vector>

#include "building_config.h"

#ifdef __USPARSE_BUILD_BACKEND_ROCM__
#include <rocsparse/rocsparse.h>
#endif
#ifndef __USPARSE_BUILD_BACKEND_ROCM__
struct rocsparse_mat_info {}; // for passing compilation, will be defined in rocsparse.h
struct rocsparse_handle{};
struct rocsparse_mat_descr{};
#endif

#include <hip/hip_runtime.h>
// #include "hip/hip_runtime_api.h"

#include "sparseMatrix.hpp"
#include "kernels/usparse_gmres.h"

using RESULT = std::tuple<int, float, double>;

void init_device_data(SpM<double> *h_A, double *h_x, SpM<double> *dev_A, double **dev_x, double **dev_y,
                      var_csr_desc<int, double> *d_csr_dest);
void init_matrix_info(SpM<double> *dev_A, rocsparse_handle *handle, rocsparse_mat_descr *descrA,
                      rocsparse_mat_info *info);

void destory_device_data(SpM<double> *dev_A, double *d_y, double *d_x);
void destory_device_matrix(double *d_r0, double *d_V, double *d_s, double *d_H, double *d_cs, double *d_sn,
                           double *d_vi);
void destory_descr_info(rocsparse_mat_info info, rocsparse_handle handle, rocsparse_mat_descr descrA);
void init_device_matrix(double **d_r0, double **d_V, double **d_s, double **d_H, double **d_cs, double **d_sn,
                        double **d_vi, uint N, int m);

// init device matrix float
void init_device_matrix_float(float **d_V_f, float **d_H_f, float **d_vi_f, float **d_sn_f, float **d_cs_f,
                              float **d_r0_f, float **d_A__val_f, float **d_s_f, float **d_b_f, uint N, int m,
                              uint nnz);

// float vector
void destory_device_float(float *d_V_f, float *d_H_f, float *d_vi_f, float *d_sn_f, float *d_cs_f, float *d_r0_f,
                          float *d_A_val_f, float *d_s_f, float *d_b_f);

void initDCU(int device_id = 0);

#endif
