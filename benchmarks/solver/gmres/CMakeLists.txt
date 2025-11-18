# set binary path and lib storage path for compiled files.
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

find_package(rocSPARSE REQUIRED)

add_executable(
    gmres-gpu-benchmark
    main.cpp
    init.cpp
    verification.cpp
    )

target_link_libraries(gmres-gpu-benchmark PRIVATE
    roc::rocsparse
    gmres-device
    hipblas
    hipsparse
    ${USPARSE_GMRES_KERNEL_LIB_NAME}
    ${USPARSE_LIB_NAME}
    )
