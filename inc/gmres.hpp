#ifndef GMRES_H
#define GMRES_H
#include <iostream>
#include <stdio.h>
#include <tuple>
#include <vector>

#include "init.hpp"
#include "sparseMatrix.hpp"
#include <hip/hip_runtime.h>
#include <hipsparse.h> // cusparseSpMV
#include <rocsparse/rocsparse.h>

using RESULT = std::tuple<int, float, double>;

void initialize(SpM<double> *A, double *x, double *b);
void sovlerTri(int Am, int i, double *H, double *s);
void rotation2(uint Am, double *H, double *cs, double *sn, double *s, uint i);
void rotation3(uint Am, double *H, double *cs, double *sn, double *s, uint i);

// rocsparse
void device_spmv(SpM<double> *dec_A, const double *dev_x, double *dev_y, rocsparse_handle *handle,
                 rocsparse_mat_descr *descrA, rocsparse_mat_info *info);

RESULT gmres(SpM<double> *A_d, double *_x, double *_b, SpM<double> *dev_A, double *dev_x, double *dev_y);
void assign2(double *a, double *s);
void fill2(uint Am, double *a, double s);

#endif
