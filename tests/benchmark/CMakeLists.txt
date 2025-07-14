set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "")
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "")
add_subdirectory(${PROJECT_SOURCE_DIR}/third-party/google-benchmark-1.8.5 gbench-1.8.5)

find_package(rocSPARSE REQUIRED)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

add_executable(
        usparse_benchmark
        usparse/usparse_level1_benchmark.cc
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
        roc::rocsparse
        ${USPARSE_KERNEL_LIB_NAME}
)
