#ifndef INITAL_H
#define INITAL_H

#include <cassert>
#include <cmath>
#include <hipsparse.h> // cusparseSpMV
#include <iostream>
#include <rocsparse/rocsparse.h>
#include <string.h>
#include <vector>

#include <hip/hip_runtime.h>
// #include "hip/hip_runtime_api.h"

#include "sparseMatrix.hpp"
#include "utils.hpp"

using RESULT = std::tuple<int, float, double>;

void init_device_data(SpM<double> *h_A, double *h_x, SpM<double> *dev_A, double **dev_x, double **dev_y);
void init_matrix_info(SpM<double> *dev_A, rocsparse_handle *handle, rocsparse_mat_descr *descrA,
                      rocsparse_mat_info *info);

void destory_device_data(SpM<double> *dev_A, double *d_y, double *d_x);
void destory_device_matrix(double *d_r0, double *d_V, double *d_s, double *d_H, double *d_cs, double *d_sn,
                           double *d_vi);
void destory_descr_info(rocsparse_mat_info info, rocsparse_handle handle, rocsparse_mat_descr descrA);
void init_device_matrix(double **d_r0, double **d_V, double **d_s, double **d_H, double **d_cs, double **d_sn,
                        double **d_vi, uint N, int m);

void initDCU(int device_id = 0);

#endif
