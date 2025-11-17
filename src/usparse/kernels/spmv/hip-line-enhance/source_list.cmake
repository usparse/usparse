set(CURRENT_ACC_HIP_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/kernels/spmv/hip-line-enhance)

set(ACC_HEADER_FILES ${ACC_HEADER_FILES}
        ${CURRENT_ACC_HIP_SOURCE_DIR}/line_enhance_spmv.h
        ${CURRENT_ACC_HIP_SOURCE_DIR}/line_enhance_spmv_imp.h
        ${CURRENT_ACC_HIP_SOURCE_DIR}/line_enhance_spmv_imp.inl
        )

set(ACC_SOURCE_FILES ${ACC_SOURCE_FILES}
        ${CURRENT_ACC_HIP_SOURCE_DIR}/line_enhance_spmv.cpp
        )
