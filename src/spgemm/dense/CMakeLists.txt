
set(ACC_HIP_HEADER_FILES ${ACC_HIP_HEADER_FILES}
        include/types.hpp
        include/spgemm_dense_kernel.hpp
        include/dense2csr.hpp
        include/dense2csr_kernel.hpp
        include/nnz_calc_kernel.hpp
        include/utils_kernel.hpp
        )
set(ACC_HIP_SOURCE_FILES ${ACC_HIP_SOURCE_FILES}
        src/spgemm_dense_kernel.cpp
        )

include_directories(
        ${ACC_HIP_HEADER_FILES}
        )

# set_source_files_properties(${ACC_HIP_SOURCE_FILES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
# hip_add_library(${SPGEMM_KERNEL_LIB_NAME} STATIC ${ACC_HIP_SOURCE_FILES} ${ACC_HIP_HEADER_FILES} HIPCC_OPTIONS ${SPGEMM_HIPCC_OPTIONS})

add_library(${SPGEMM_DENSE_KERNEL_LIB_NAME} STATIC ${ACC_HIP_SOURCE_FILES} ${ACC_HIP_HEADER_FILES})
target_include_directories(
        ${SPGEMM_DENSE_KERNEL_LIB_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}$<SEMICOLON>${PROJECT_BINARY_DIR}/generated>
        $<INSTALL_INTERFACE:include>
)
