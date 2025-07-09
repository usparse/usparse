cmake_minimum_required(VERSION 3.10)
project(gmres-hip)

set(CMAKE_CXX_STANDARD 11)

include(config.cmake)
include(dep.cmake)

add_subdirectory(library)
add_subdirectory(src)
