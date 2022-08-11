# this file is include in other directory, current path is ${ACC_SRC_PATH}.
set(CURRENT_ACC_HIP_SOURCE_DIR ${ACC_SRC_PATH})

set(ACC_HEADER_FILES ${ACC_HEADER_FILES}
        ${CURRENT_ACC_HIP_SOURCE_DIR}/dense_impl.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/dense_kernel.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/dense2csr_kernel.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/dense2csr_impl.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/nnz_kernel.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/nnz_impl.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/common.hpp
        )
