# set binary path and lib storage path for compiled files.
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

if (USPARSE_GPU_BACKEND_CUDA)
    set(BENCHMARK_DEP_LIBS
        ${BENCHMARK_DEP_LIBS}
        ${USPARSE_GPU_BACKEND_SPARSE_LIB}
        cublas
        )
endif()

if (USPARSE_GPU_BACKEND_ROCM)
    set(BENCHMARK_DEP_LIBS
        ${BENCHMARK_DEP_LIBS}
        ${USPARSE_GPU_BACKEND_SPARSE_LIB}
        hipblas
        hipsparse
        )
endif ()

set_source_files_properties_hip(main.cpp init.cpp)

add_executable(
    gmres-gpu-benchmark
    main.cpp
    init.cpp
    verification.cpp
    )

target_link_libraries(gmres-gpu-benchmark PRIVATE
    ${BENCHMARK_DEP_LIBS}
    gmres-device
    ${USPARSE_GMRES_KERNEL_LIB_NAME}
    ${USPARSE_LIB_NAME}
    )
