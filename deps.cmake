string(TOUPPER "${USPARSE_GPU_BACKEND}" USPARSE_GPU_BACKEND_UPPER)

# find HIP if HIP is enabled, and embeded UPTK is not enabled.
if (USPARSE_ENABLE_HIP)
    if(${USPARSE_GPU_BACKEND_UPPER} STREQUAL ROCM)
        include(cmake/use_backend_rocm.cmake)
        set(USPARSE_USE_EMBEDED_UPTK OFF)
    elseif(${USPARSE_GPU_BACKEND_UPPER} STREQUAL CUDA)
        include(cmake/use_backend_cuda.cmake)
        # use the embeded UPTK implementation in usparse for running usparse on CUDA env, instead of HIP.
        set(USPARSE_USE_EMBEDED_UPTK ON)
    else()
        message(FATAL_ERROR "Unsupported GPU_BACKEND: ${USPARSE_GPU_BACKEND}")
    endif()

    if(USPARSE_USE_EMBEDED_UPTK)
        include_directories(${PROJECT_SOURCE_DIR}/library/uptk)
    endif()
endif ()
