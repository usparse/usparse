# set binary path and lib storage path for compiling process.
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

# set variable ACC_FILES for architecture source files.
# set variable ACC_LIBS to link if some architecture is enabled.
set(SPGEMM_KERNEL_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src/spgemm)

include(source_list.cmake)

set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

# make binary
#add_executable(${SPGEMM_BIN_NAME} ${SOURCE_FILES})
add_executable(${SPGEMM_BENCHMARK_NAME} ${SOURCE_FILES})

target_include_directories(
#        ${SPGEMM_BIN_NAME}
        ${SPGEMM_BENCHMARK_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}$<SEMICOLON>${PROJECT_BINARY_DIR}/generated>
        $<INSTALL_INTERFACE:include>
        PRIVATE
        ${SPGEMM_KERNEL_SOURCE_DIR}
)

message(STATUS "ACC_LIBS: "${ACC_LIBS})

target_link_libraries(
#        ${SPGEMM_BIN_NAME}
        ${SPGEMM_BENCHMARK_NAME}
        PUBLIC
        ${ACC_LIBS}
        rocsparse # todo: use rocsparse.
)
