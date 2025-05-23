cmake_minimum_required(VERSION 3.15)
project(usparse)

set(CMAKE_CXX_STANDARD 14)

include(config.cmake)
include(deps.cmake)

add_subdirectory(library)
add_subdirectory(benchmarks)
