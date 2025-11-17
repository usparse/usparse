cmake_minimum_required(VERSION 3.21)
project(usparse)

set(CMAKE_CXX_STANDARD 14)

include(config.cmake)
include(deps.cmake)

if(USPARSE_BUILD_DEP_BY_PKG)
    include(pkg.dep.cmake)
else()
    include_directories(vendor/include)
endif()

add_subdirectory(library)
add_subdirectory(cli)

if(USPARSE_BUILD_BENCHMARK)
    add_subdirectory(benchmarks) # usparse benchmark
endif()

if(SPMV_BUILD_TOOLS)
    add_subdirectory(tools)
endif()
# gmres-gpu start
include(config.cmake)
include(dep.cmake)

add_subdirectory(library)
add_subdirectory(src)

if (GMRES_TOOLS_ENABLE)
    add_subdirectory(tools)
endif ()

add_subdirectory(tests)
# gmres-gpu end
