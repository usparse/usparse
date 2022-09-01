# this file is include in other directory, current path is ${ACC_SRC_PATH}.
set(CURRENT_ACC_HIP_SOURCE_DIR ${ACC_SRC_PATH})

set(ACC_HEADER_FILES ${ACC_HEADER_FILES}
        ${CURRENT_ACC_HIP_SOURCE_DIR}/compress_matrix.h
        ${CURRENT_ACC_HIP_SOURCE_DIR}/build_flag.h
        ${CURRENT_ACC_HIP_SOURCE_DIR}/csr_matrix.h
        ${CURRENT_ACC_HIP_SOURCE_DIR}/error_macro.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/timer.h
        ${CURRENT_ACC_HIP_SOURCE_DIR}/hash.h
        ${CURRENT_ACC_HIP_SOURCE_DIR}/hash_map.h
        ${CURRENT_ACC_HIP_SOURCE_DIR}/hash_map_insert.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/spgemm.h
        ${CURRENT_ACC_HIP_SOURCE_DIR}/spgemm_core.h
        ${CURRENT_ACC_HIP_SOURCE_DIR}/spgemm_core_impl.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/spgemm_numeric.h
        ${CURRENT_ACC_HIP_SOURCE_DIR}/spgemm_numeric_impl.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/spgemm_numeric_kernel.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/spgemm_symbolic.h
        ${CURRENT_ACC_HIP_SOURCE_DIR}/spgemm_symbolic_impl.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/spgemm_symbolic_kernel.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/spgemm_compress.h
        ${CURRENT_ACC_HIP_SOURCE_DIR}/spgemm_compress_impl.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/spgemm_compress_kernel.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/scan.h
        ${CURRENT_ACC_HIP_SOURCE_DIR}/scan_impl.hpp
        # ${CURRENT_ACC_HIP_SOURCE_DIR}/scan_kernel.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/split_config.h
        ${CURRENT_ACC_HIP_SOURCE_DIR}/split_group.h
        ${CURRENT_ACC_HIP_SOURCE_DIR}/split_group_impl.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/split_group_kernel.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/max_num.h
        ${CURRENT_ACC_HIP_SOURCE_DIR}/max_num_impl.hpp
        ${CURRENT_ACC_HIP_SOURCE_DIR}/max_num_kernel.hpp
        )
