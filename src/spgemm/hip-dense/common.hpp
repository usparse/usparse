#ifndef SPGEMM_DENSE_TYPES_H
#define SPGEMM_DENSE_TYPES_H

#include <hip/hip_runtime.h>
#include <hipsparse.h>
#include <rocsparse.h>


#define HIP_CHECK(stat)                                                                        \
{                                                                                              \
    if(stat != hipSuccess)                                                                     \
    {                                                                                          \
        std::cerr << "Error: hip error " << stat <<" in line " << __LINE__ << std::endl;       \
        exit(-1);                                                                              \
    }                                                                                          \
}

#endif //SPGEMM_DENSE_TYPES_H
