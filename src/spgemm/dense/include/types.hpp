#pragma once

#include <hip/hip_runtime.h>
#include <hipsparse.h>
#include <rocsparse.h>

typedef double xtype;
typedef double csrValueType;
typedef int csrIndexType;

enum dense_matrix_storage_option{row_first=1,col_first=0};

#define HIP_CHECK(stat)                                                                        \
{                                                                                              \
    if(stat != hipSuccess)                                                                     \
    {                                                                                          \
        std::cerr << "Error: hip error " << stat <<" in line " << __LINE__ << std::endl;       \
        exit(-1);                                                                              \
    }                                                                                          \
}
