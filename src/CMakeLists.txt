include(configure.cmake)

set(HEADER_FILES
        common_function.hpp
        Csrsparse.hpp
        )

set(SOURCE_FILES
        main.cpp
        )

# set binary path and lib storage path for compiling process.
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

include_directories(
        ${HEADER_FILES}
)


# set variable ACC_FILES for architecture source files.
# set variable ACC_LIBS to link if some architecture is enabled.
set(SPGEMM_KERNEL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/spgemm)
add_subdirectory(spgemm)

# make lib
add_executable(${SPGEMM_BIN_NAME} ${HEADER_FILES} ${SOURCE_FILES})
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
