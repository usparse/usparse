set(CURRENT_ACC_HIP_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/kernels/spmv/hip-vector-row)

set(ACC_HEADER_FILES ${ACC_HEADER_FILES}
        ${CURRENT_ACC_HIP_SOURCE_DIR}/vector_config.h
        ${CURRENT_ACC_HIP_SOURCE_DIR}/vector_row.h
        ${CURRENT_ACC_HIP_SOURCE_DIR}/vector_row.inl
        ${CURRENT_ACC_HIP_SOURCE_DIR}/vector_row_adaptive.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/vector_row_native.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/opt_double_buffer.hpp
        )

set(ACC_SOURCE_FILES ${ACC_SOURCE_FILES}
        ${CURRENT_ACC_HIP_SOURCE_DIR}/vector_row.cpp
        )
