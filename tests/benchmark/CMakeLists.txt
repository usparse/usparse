set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "")
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "")
add_subdirectory(${PROJECT_SOURCE_DIR}/third-party/google-benchmark-1.9.4 gbench-1.9.4)
if(TARGET benchmark AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(benchmark PRIVATE -Wno-return-type)
endif()

set(BENCHMARK_DEP_LIBS ${BENCHMARK_DEP_LIBS} ${USPARSE_GPU_BACKEND_SPARSE_LIB})

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

set_source_files_properties_hip(
        usparse/usparse_level1_benchmark.cc
        usparse/usparse_level1_dotprof_axpy_fusion.cc)
add_executable(
        usparse_benchmark
        usparse/usparse_level1_benchmark.cc
        usparse/usparse_level1_dotprof_axpy_fusion.cc
)
target_include_directories(
        usparse_benchmark
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(
        usparse_benchmark
        PUBLIC
        benchmark::benchmark
        ${BENCHMARK_DEP_LIBS}
        ${USPARSE_GMRES_KERNEL_LIB_NAME}
        ${USPARSE_LIB_NAME}
)

set_source_files_properties_hip(usparse/usparse_level2_csrmv_benchmark.cc)
add_executable(
        ustbSparse_spmv_benchmark
        usparse/usparse_level2_csrmv_benchmark.cc
)
target_include_directories(
        ustbSparse_spmv_benchmark
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(
        ustbSparse_spmv_benchmark
        PUBLIC
        benchmark::benchmark
        ${BENCHMARK_DEP_LIBS}
        ${USPARSE_GMRES_KERNEL_LIB_NAME}
        ${USPARSE_LIB_NAME}
)
