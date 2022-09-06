# this file is include in other directory, current path is ${ACC_SRC_PATH}.
set(CURRENT_ACC_HIP_SOURCE_DIR ${ACC_SRC_PATH})

set(ACC_HEADER_FILES ${ACC_HEADER_FILES}
        ${CURRENT_ACC_HIP_SOURCE_DIR}/csr_sort.h
        ${CURRENT_ACC_HIP_SOURCE_DIR}/block_radix_sort.h
        )
