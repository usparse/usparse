# all enabled strategies
set(ENABLED_STRATEGIES
        adaptive
        flat
        light
        line
        thread_row
        vector_row
        line_enhance
        csr_adaptive_plus
        )

# to include strategies source files.
set(CURRENT_ACC_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
foreach (X_STRATEGY IN LISTS ENABLED_STRATEGIES)
    string(TOLOWER ${X_STRATEGY} X_STRATEGY_LOWER)

    # set directory postfix for strategy source files.
    if (X_STRATEGY_LOWER MATCHES "default")
        set(X_KERNEL_STRATEGY_SRC_DIR_POSTFIX "")
    else ()
        string(REPLACE "_" "-" X_KERNEL_STRATEGY_SRC_DIR_POSTFIX "-${X_STRATEGY_LOWER}")
    endif ()

    set(ACC_SRC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${ARCH_NAME}${X_KERNEL_STRATEGY_SRC_DIR_POSTFIX}")

    # check ACC_SRC_PATH and add_subdirectory
    if (NOT EXISTS "${ACC_SRC_PATH}" OR NOT IS_DIRECTORY "${ACC_SRC_PATH}")
        message(FATAL_ERROR "Architecture source files directory not found: ${ACC_SRC_PATH}")
    else ()
        MESSAGE(STATUS "Add arch source files in ${ACC_SRC_PATH}")
        include(${ACC_SRC_PATH}/source_list.cmake) # set ${ACC_HEADER_FILES} and ${ACC_SOURCE_FILES} here.
    endif ()
endforeach ()

set(ACC_HEADER_FILES ${ACC_HEADER_FILES}
        ${USPARSE_HEADER_PATH}/usparse_matrix.h
        ${USPARSE_API_PATH}/spmv.h
        )
set(ACC_SOURCE_FILES ${ACC_SOURCE_FILES} ${USPARSE_API_PATH}/spmv_imp.cpp strategy_picker.cpp)

# build static lib
if (USPARSE_ENABLE_HIP)
    set_source_files_properties(${ACC_SOURCE_FILES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
    hip_add_library(${SPMV_KERNEL_LIB_NAME} STATIC ${ACC_SOURCE_FILES} ${ACC_HEADER_FILES} HIPCC_OPTIONS ${USPARSE_HIPCC_OPTIONS})
    target_include_directories(
            ${SPMV_KERNEL_LIB_NAME}
            PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}$<SEMICOLON>${USPARSE_API_PATH}/../$<SEMICOLON>${USPARSE_HEADER_PATH}$<SEMICOLON>${USPARSE_HIP_SOURCE_PATH}$<SEMICOLON>${PROJECT_BINARY_DIR}/generated>
            $<INSTALL_INTERFACE:include>
    )
endif ()

##########################
##### install files
##########################
# install bin and libs
install(TARGETS ${SPMV_KERNEL_LIB_NAME}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        )

# install header.
install(DIRECTORY "./"
        DESTINATION "include"
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp"
        PATTERN "*.inl"
        )
