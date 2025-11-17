#ifndef VERIFICATION_H
#define VERIFICATION_H
#include <iostream>
#include <stdio.h>
#include <tuple>
#include <vector>

#include "init.hpp"
#include "sparseMatrix.hpp"
#include <hip/hip_runtime.h>

double Verification(SpM<double> *d_A, double *x, double *b, double *r0, int N);

#endif
