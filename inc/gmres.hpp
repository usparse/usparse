#ifndef GMRES_H
#define GMRES_H
#include <iostream>
#include <stdio.h>
#include <tuple>
#include <vector>

#include "init.hpp"
#include "sparseMatrix.hpp"
#include "verification.hpp"
#include <hip/hip_runtime.h>

using RESULT = std::tuple<int, float, double>;

void initialize(SpM<double> *A, double *x, double *b);
void sovlerTri(int Am, int i, double *H, double *s);
void rotation2(uint Am, double *H, double *cs, double *sn, double *s, uint i);
void rotation3(uint Am, double *H, double *cs, double *sn, double *s, uint i);

RESULT gmres(SpM<double> *A_d, double *_x, double *_b, SpM<double> *dev_A, double *dev_x, double *dev_y,
             csr_desc<int, double> d_csr_desc);
RESULT gmres_float(SpM<double> *A_d, double *_x, double *_b, SpM<double> *dev_A, double *dev_x, double *dev_y,
             csr_desc<int, double> d_csr_desc);
void assign2(double *a, double *s);
void fill2(uint Am, double *a, double s);

#endif
