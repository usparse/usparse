# find HIP
if (USPARSE_ENABLE_HIP)
    # see https://github.com/ROCm-Developer-Tools/HIP/blob/master/samples/2_Cookbook/12_cmake_hip_add_executable
    if (NOT DEFINED HIP_PATH)
        if (NOT DEFINED ENV{HIP_PATH})
            set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
        else ()
            set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
        endif ()
    endif ()
    set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})

    find_package(HIP REQUIRED)
    if (HIP_FOUND)
        message(STATUS "Found HIP: " ${HIP_VERSION})
    else ()
        message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location.")
    endif ()
    set(USPARSE_HIPCC_OPTIONS) # set hipcc arguments
endif ()

# csrmv or spmv-acc
set(USPARSER_CSRMV_HEADER_FILES
        ${USPARSE_HEADER_PATH}/usparse_matrix.h
        ${USPARSE_API_PATH}/spmv.h
        level2/csrmv_kernels/adaptive/adaptive.h
        level2/csrmv_kernels/csr-adaptive-plus/csr_adaptive_plus_spmv.h
        level2/csrmv_kernels/csr-adaptive-plus/csr_adaptive_plus_spmv_imp.h
        level2/csrmv_kernels/flat/spmv_hip_acc_imp.h
        level2/csrmv_kernels/flat/flat_config.h
        level2/csrmv_kernels/flat/flat_imp.inl
        level2/csrmv_kernels/flat/flat_imp_one_pass.hpp
        level2/csrmv_kernels/flat/flat_reduce.hpp
        level2/csrmv_kernels/light/spmv_hip_acc_imp.h
        level2/csrmv_kernels/light/spmv_hip_acc_imp.inl
        level2/csrmv_kernels/line/line_imp_one_pass.inl
        level2/csrmv_kernels/line/line_kernel_imp.inl
        level2/csrmv_kernels/line/line_adaptive_one_pass.inl
        level2/csrmv_kernels/line/line_strategy.h
        level2/csrmv_kernels/line/line_config.h
        level2/csrmv_kernels/line-enhance/csrmv_line_enhance.h
        level2/csrmv_kernels/line-enhance/line_enhance_kernels_imp.h
        level2/csrmv_kernels/line-enhance/line_enhance_kernels_imp.inl
        level2/csrmv_kernels/thread-row/thread_row.h
        level2/csrmv_kernels/thread-row/thread_row.inl
        level2/csrmv_kernels/thread-row/thread_row_config.h
        level2/csrmv_kernels/vector-row/vector_config.h
        level2/csrmv_kernels/vector-row/vector_row.h
        level2/csrmv_kernels/vector-row/vector_row.inl
        level2/csrmv_kernels/vector-row/vector_row_adaptive.hpp
        level2/csrmv_kernels/vector-row/vector_row_native.hpp
        level2/csrmv_kernels/vector-row/opt_double_buffer.hpp
        )
        
set(USPARSER_CSRMV_SOURCE_FILES
        ${USPARSE_API_PATH}/spmv_imp.cpp
        level2/csrmv.cpp
        level2/csrmv_kernels/strategy_picker.cpp
        level2/csrmv_kernels/adaptive/adaptive.cpp
        level2/csrmv_kernels/csr-adaptive-plus/csr_adaptive_plus_spmv.cpp
        level2/csrmv_kernels/csr-adaptive-plus/csr_adaptive_plus_analyze.cpp
        level2/csrmv_kernels/flat/flat.cpp
        level2/csrmv_kernels/light/light_spmv.cpp
        level2/csrmv_kernels/line/line_strategy.cpp
        level2/csrmv_kernels/line-enhance/csrmv_line_enhance.cpp
        level2/csrmv_kernels/thread-row/thread_row.cpp
        level2/csrmv_kernels/thread-row/native_thread_row.cpp
        level2/csrmv_kernels/vector-row/vector_row.cpp
        )

# spgemm
add_subdirectory(level3/)

set(USPARSE_HEADER_FILES ${USPARSE_HEADER_FILES}
        ${USPARSER_CSRMV_HEADER_FILES}
        ${USPARSE_HEADER_PATH}/usparse.h
        ${USPARSE_HEADER_PATH}/usparse_matrix.h
        ${USPARSE_HEADER_PATH}/usparse_types.h
        conversion/coo2csr.h
        level1/usparse_blas_level1.h
        level2/coomv_kernels/coomv_flat_kernel.h
        level2/coomv_kernels/coomv_spmv_acc_wrappers.h
        )

set(USPARSE_SOURCE_FILES ${USPARSE_SOURCE_FILES}
        ${USPARSER_CSRMV_SOURCE_FILES}
        conversion/coo2csr.cpp
        level1/usparse_blas_level1.cpp
        level2/coomv_kernels/coomv_flat_kernel.cc
        level2/coomv_kernels/coomv_spmv_acc_wrappers.cc
        level3/csrsm_kernels.cpp
        level3/coomm_kernels/coomm_conversion_based.cc
        level3/csrmm_kernels/csrmm_kernel_rowsplit.cpp
        ../internal/usparse_internal_spmv.cc
        ../internal/usparse_internal_spmm.cc
        )

add_subdirectory(solver/gmres)

# build static lib
set_source_files_properties(${USPARSE_SOURCE_FILES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_library(${USPARSE_LIB_NAME} STATIC ${USPARSE_SOURCE_FILES} ${USPARSE_HEADER_FILES} HIPCC_OPTIONS ${USPARSE_HIPCC_OPTIONS})
target_include_directories(
        ${USPARSE_LIB_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}$<SEMICOLON>${USPARSE_HEADER_PATH}$<SEMICOLON>${USPARSE_HEADER_PATH}/../$<SEMICOLON>${PROJECT_BINARY_DIR}/generated>
        $<INSTALL_INTERFACE:include>
)

##########################
##### install files
##########################
# install bin and libs
install(TARGETS ${USPARSE_LIB_NAME}
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
