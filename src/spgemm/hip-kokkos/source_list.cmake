# this file is include in other directory, current path is ${ACC_SRC_PATH}.
set(CURRENT_ACC_HIP_SOURCE_DIR ${ACC_SRC_PATH})

set(ACC_HEADER_FILES ${ACC_HEADER_FILES}
        ${CURRENT_ACC_HIP_SOURCE_DIR}/bitset.h
        ${CURRENT_ACC_HIP_SOURCE_DIR}/build_flag.h
        ${CURRENT_ACC_HIP_SOURCE_DIR}/csr_matrix.h
        ${CURRENT_ACC_HIP_SOURCE_DIR}/error_macro.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/hash_map.h
        ${CURRENT_ACC_HIP_SOURCE_DIR}/spgemm_core.h
        ${CURRENT_ACC_HIP_SOURCE_DIR}/spgemm_core.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/spgemm_kernel.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/spgemm.h
        ${CURRENT_ACC_HIP_SOURCE_DIR}/utils_kernel.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/dense_to_sparse.hpp
        )
