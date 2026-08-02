/**
 * Created by genshen on 2026/04/28.
 */
#ifndef USPARSE_HIP_RUNTIME_API_H
#define USPARSE_HIP_RUNTIME_API_H

#include "building_config.h"

#ifdef __USPARSE_BUILD_BACKEND_CUDA__
#include "cuda/cuda_runtime_api_map.h"
#endif

#ifdef __USPARSE_BUILD_BACKEND_MUSA__
#include "musa/musa_runtime_api_map.h"
#endif

#endif // USPARSE_HIP_RUNTIME_API_H
