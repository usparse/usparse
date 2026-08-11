#ifndef VERIFICATION_H
#define VERIFICATION_H

#include <hip/hip_runtime.h>
#include <iostream>
#include <stdio.h>
#include <tuple>
#include <vector>

#include <usparse_matrix.h>

#include "init.hpp"

double Verification(const usparse::csr_mat_descr<double, int> d_A, double *x, double *b, double *r0, int N);

#endif
