//
// Created by genshen to 2025-07-09.
//

#ifndef USPARSE_KERNELS_LEVEL2_USPARSE_GMRES_H
#define USPARSE_KERNELS_LEVEL2_USPARSE_GMRES_H

// header in usparse library for GMRES calling

#include "kernels/gmres_optimization_config.h"

#include "kernels/gmres-big-fusion/gmres_big_fusion.hpp"
#include "kernels/level1/usparse_dotprod_axpy_fusion.h"
#include "kernels/level2/usparse_csrmv.h"
#include "kernels/level2/usparse_rotation.h"
#include "kernels/level2/usparse_solver_tri.h"
#include "common/usparse_utils.hpp"

#endif // USPARSE_KERNELS_LEVEL2_USPARSE_GMRES_H
