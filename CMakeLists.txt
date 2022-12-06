cmake_minimum_required(VERSION 3.15)
project(spgemm)

set(CMAKE_CXX_STANDARD 14)

include(config.cmake)

# set binary path and lib storage path for compiling process.
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)


# set variable ACC_FILES for architecture source files.
# set variable ACC_LIBS to link if some architecture is enabled.
set(SPGEMM_KERNEL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/spgemm)

set(SOURCE_FILES
        main.cpp
)

include(${CMAKE_CURRENT_SOURCE_DIR}/tools/source_list.cmake)

add_subdirectory(src)

# make lib
add_executable(${SPGEMM_BIN_NAME} ${SOURCE_FILES})
target_include_directories(
        ${SPGEMM_BIN_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}$<SEMICOLON>${PROJECT_BINARY_DIR}/generated>
        $<INSTALL_INTERFACE:include>
        PRIVATE
        ${SPGEMM_KERNEL_SOURCE_DIR}
)

target_link_libraries(
        ${SPGEMM_BIN_NAME}
        PUBLIC
        ${ACC_LIBS}
)