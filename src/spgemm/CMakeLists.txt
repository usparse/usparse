# all enabled strategies
set(ENABLED_STRATEGIES
    kokkos
    dense
    )

# find HIP

# to include strategies source files.
set(CURRENT_SPGEMM_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

foreach (X_STRATEGY IN LISTS ENABLED_STRATEGIES)
    string(TOLOWER ${X_STRATEGY} X_STRATEGY_LOWER)

    # set directory postfix for strategy source files.
    if (X_STRATEGY_LOWER MATCHES "default")
        set(X_KERNEL_STRATEGY_SRC_DIR_POSTFIX "")
    else ()
        string(REPLACE "_" "-" X_KERNEL_STRATEGY_SRC_DIR_POSTFIX "-${X_STRATEGY_LOWER}")
    endif ()
    set (ACC_SRC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${ARCH_NAME}${X_KERNEL_STRATEGY_SRC_DIR_POSTFIX}")
    # check ACC_SRC_PATH and add_subdirectory
    if (NOT EXISTS "${ACC_SRC_PATH}" OR NOT IS_DIRECTORY "${ACC_SRC_PATH}")
        MESSAGE(FATAL_ERROR "Architecture source files directory not found: ${ACC_SRC_PATH}")
    else ()
        MESSAGE(STATUS "Add arch source files in ${ACC_SRC_PATH}")
        # set ${ACC_HEADER_FILES} and ${ACC_SOURCE_FILES} here.
        include(${ACC_SRC_PATH}/source_list.cmake)
    endif ()
endforeach ()

set(ACC_HIP_HEADER_FILES ${ACC_HEADER_FILES}
        api/spgemm_hip_api.h
        api/spgemm_hip_types.h
        common/reduction_kernel.hpp
        common/scan_kernel.hpp
        )

set(ACC_HIP_SOURCE_FILES ${ACC_SOURCE_FILES}
        api/spgemm_hip_api.cc
        )

# set_source_files_properties(${ACC_HIP_SOURCE_FILES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
# hip_add_library(${SPGEMM_KERNEL_LIB_NAME} STATIC ${ACC_HIP_SOURCE_FILES} ${ACC_HIP_HEADER_FILES} HIPCC_OPTIONS ${SPGEMM_HIPCC_OPTIONS})

# build static library
MESSAGE(STATUS "sort source dir is ${SORT_SOURCE_DIR}")
add_library(${SPGEMM_KERNEL_LIB_NAME} STATIC ${ACC_HIP_SOURCE_FILES} ${ACC_HIP_HEADER_FILES})
target_include_directories(
        ${SPGEMM_KERNEL_LIB_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}$<SEMICOLON>${PROJECT_BINARY_DIR}/generated>
        $<INSTALL_INTERFACE:include>
        PRIVATE
        ${SORT_KERNEL_SOURCE_DIR}
)
target_link_libraries(
        ${SPGEMM_KERNEL_LIB_NAME}
        PRIVATE
        ${SORT_KERNEL_LIB_NAME}
)
