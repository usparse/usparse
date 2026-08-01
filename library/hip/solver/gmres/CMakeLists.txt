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

# build static lib
set(USPARSE_GMRES_HEADER_FILES ${USPARSE_GMRES_HEADER_FILES}
        kernels/gmres_utils.h
        kernels/usparse_gmres.h
        kernels/gmres-big-fusion/gmres_big_fusion.h
        kernels/level1/usparse_level1_batched.h
        kernels/level2/usparse_csrmv.h
        kernels/level2/usparse_rotation.h
        kernels/level2/usparse_solver_tri.h

)
set(USPARSE_GMRES_SOURCE_FILES ${USPARSE_GMRES_SOURCE_FILES}
        kernels/gmres_utils.cpp
        kernels/gmres-big-fusion/gmres_big_fusion.cpp
        kernels/level1/usparse_dotprod_axpy_fusion.cpp
        kernels/level1/usparse_level1_batched.cpp
        kernels/level2/usparse_csrmv.cpp
        kernels/level2/usparse_rotation.cpp
        kernels/level2/usparse_solver_tri.cpp
)

set_source_files_properties_hip(${USPARSE_GMRES_SOURCE_FILES})
hip_add_library(${USPARSE_GMRES_KERNEL_LIB_NAME} STATIC ${USPARSE_GMRES_SOURCE_FILES} ${USPARSE_GMRES_HEADER_FILES} HIPCC_OPTIONS ${USPARSE_LIB_COMPILE_OPTIONS})

message(STATUS ${USPARSE_GMRES_KERNEL_LIB_NAME})

target_include_directories(
        ${USPARSE_GMRES_KERNEL_LIB_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}$<SEMICOLON>${USPARSE_HEADER_PATH}/$<SEMICOLON>${USPARSE_HEADER_PATH}/../$<SEMICOLON>${USPARSE_HEADER_PATH}/../hip/$<SEMICOLON>${PROJECT_BINARY_DIR}/generated>
        $<INSTALL_INTERFACE:include>
)

set_source_files_properties_hip(precision_patch.cpp gmres.cpp)
add_library(
    gmres-device
    precision_patch.cpp
    gmres.cpp
    )

target_link_libraries(gmres-device PRIVATE
    ${BENCHMARK_DEP_LIBS}
    ${USPARSE_GMRES_KERNEL_LIB_NAME}
    )


##########################
##### install files
##########################
# install bin and libs
install(TARGETS gmres-device ${USPARSE_GMRES_KERNEL_LIB_NAME}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        )

# install header.
install(DIRECTORY "./"
        DESTINATION "include"
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp"
        PATTERN "*.inl"
        )
