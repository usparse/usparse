set(BENCHMARK_DEP_LIBS ${SPMV_CLI_LIB} ${USPARSE_LIB_NAME} ${USPARSE_BENCHMARK_UTILS_LIB})

if (BENCHMARK_CUDA_ENABLE_FLAG)
    set(BENCHMARK_DEP_LIBS
            ${BENCHMARK_DEP_LIBS}
            cudart
            )
endif ()

set(BENCHMARK_DEP_LIBS ${BENCHMARK_DEP_LIBS} ${USPARSE_GPU_BACKEND_SPARSE_LIB})

set(HEADER_FILES
        coomm_statistics_logger.hpp
        benchmark_coomm.h
        coo_spmm.hpp
        ../utils/benchmark_main.hpp
        )

set(SOURCE_FILES
        main.cc
        benchmark_usparse_coomm.cc
        ../utils/ubench_sparse_matrix_utils.cpp
        ../utils/ubench_vector_utils.cpp
        )
# set rocsparse benchmark if it is on ROCm
if (NOT BENCHMARK_CUDA_ENABLE_FLAG)
        set(SOURCE_FILES ${SOURCE_FILES} benchmark_rocsparse_coomm.cc)
endif()

set(COOMM_BENCHMARK_NAME coomm-gpu-benchmark)

# set binary path and lib storage path for compiling process.
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

add_executable(${COOMM_BENCHMARK_NAME} ${HEADER_FILES} ${SOURCE_FILES})
target_include_directories(
        ${COOMM_BENCHMARK_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}$<SEMICOLON>${PROJECT_BINARY_DIR}/generated>
        $<INSTALL_INTERFACE:include>
)
set_property(TARGET ${COOMM_BENCHMARK_NAME} PROPERTY CXX_STANDARD 17)

if(BENCHMARK_FORCE_SYNC_KERNELS)
        target_compile_definitions(${COOMM_BENCHMARK_NAME} PUBLIC MACRO_BENCHMARK_FORCE_KERNEL_SYNC=1)
        message(STATUS "force kernel sync for target ${COOMM_BENCHMARK_NAME} is enabled")
else()
        message(STATUS "force kernel sync for target ${COOMM_BENCHMARK_NAME} is disabled")
endif()

target_link_libraries(
        ${COOMM_BENCHMARK_NAME}
        PUBLIC
        ${BENCHMARK_DEP_LIBS}
)

# See https://cliutils.gitlab.io/modern-cmake/chapters/packages/OpenMP.html
if (SPMM_OMP_ENABLED_FLAG)
    find_package(OpenMP)
    if (${OPENMP_FOUND} AND ${OpenMP_CXX_FOUND})
        set_target_properties(${COOMM_BENCHMARK_NAME} PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
        set_target_properties(${COOMM_BENCHMARK_NAME} PROPERTIES LINK_FLAGS ${OpenMP_CXX_FLAGS})
    elseif (${OPENMP_FOUND} OR ${OpenMP_C_FOUND})
        set_target_properties(${COOMM_BENCHMARK_NAME} PROPERTIES COMPILE_FLAGS ${OpenMP_C_FLAGS})
        set_target_properties(${COOMM_BENCHMARK_NAME} PROPERTIES LINK_FLAGS ${OpenMP_C_FLAGS})
    else ()
        message(FATAL_ERROR "OpenMP not found, but required OpenMP support.")
    endif ()
    message(STATUS "Build ${COOMM_BENCHMARK_NAME} with OpenMP support.")
endif ()

##########################
##### install files
##########################
# install bin and libs
install(TARGETS ${COOMM_BENCHMARK_NAME} ${SPMV_BENCHMARK_CUB_LIB} ${SPMV_BENCHMARK_HOLA_LIB} ${USPARSE_BENCHMARK_UTILS_LIB} ${SPMV_BENCHMARK_HOLA_HIP_LIB}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        )
