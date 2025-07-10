cmake_minimum_required(VERSION 3.21)
project(usparse)

set(CMAKE_CXX_STANDARD 14)

include(config.cmake)
include(deps.cmake)
include(pkg.dep.cmake)

add_subdirectory(library)
add_subdirectory(cli)
# add_subdirectory(src) # spgemm src.

if(USPARSE_BUILD_BENCHMARK)
    add_subdirectory(benchmarks) # usparse benchmark
    # add_subdirectory(benchmark) # spgemm benchmark
endif()

if(SPMV_BUILD_TOOLS)
    add_subdirectory(tools)
endif()
