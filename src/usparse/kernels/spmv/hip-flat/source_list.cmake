set(CURRENT_ACC_HIP_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/kernels/spmv/hip-flat)

set(ACC_HEADER_FILES ${ACC_HEADER_FILES}
        ${CURRENT_ACC_HIP_SOURCE_DIR}/spmv_hip_acc_imp.h
        ${CURRENT_ACC_HIP_SOURCE_DIR}/flat_config.h
        ${CURRENT_ACC_HIP_SOURCE_DIR}/flat_imp.inl
        ${CURRENT_ACC_HIP_SOURCE_DIR}/flat_imp_one_pass.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/flat_reduce.hpp
        )

set(ACC_SOURCE_FILES ${ACC_SOURCE_FILES}
        ${CURRENT_ACC_HIP_SOURCE_DIR}/flat.cpp
        )
