# this file is include in other directory, current path is ${ACC_SRC_PATH}.
set(CURRENT_ACC_HIP_SOURCE_DIR ${ACC_SRC_PATH})

set(ACC_HEADER_FILES ${ACC_HEADER_FILES}
        ${CURRENT_ACC_HIP_SOURCE_DIR}/compress_matrix.h
        ${CURRENT_ACC_HIP_SOURCE_DIR}/csr_matrix.h
        ${CURRENT_ACC_HIP_SOURCE_DIR}/align.h
        ${CURRENT_ACC_HIP_SOURCE_DIR}/hash_map_insert_kernel.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/spgemm.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/spgemm_core_impl.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/spgemm_numeric_impl.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/spgemm_numeric_kernel.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/spgemm_symbolic_impl.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/spgemm_symbolic_kernel.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/spgemm_compress_impl.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/spgemm_compress_kernel.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/split_config.h
        ${CURRENT_ACC_HIP_SOURCE_DIR}/split_group_impl.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/split_group_kernel.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/create_bp_impl.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/create_bp_kernel.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/spgemm_buffer_impl.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/spgemm_adapt_policy.hpp
        )
