# this file is include in other directory, current path is ${COMMON_SRC_PATH}.
set(CURRENT_COMMON_SOURCE_DIR ${COMMON_SRC_PATH})

set(COMMON_HEADER_FILES ${COMMON_HEADER_FILES}
        ${CURRENT_COMMON_SOURCE_DIR}/common.h
        ${CURRENT_COMMON_SOURCE_DIR}/scan/scan_impl.hpp
        ${CURRENT_COMMON_SOURCE_DIR}/scan/scan_in_device.hpp
        ${CURRENT_COMMON_SOURCE_DIR}/scan/reduce_then_scan.hpp
        ${CURRENT_COMMON_SOURCE_DIR}/scan/lookback_scan.hpp
        ${CURRENT_COMMON_SOURCE_DIR}/scan/single_block_scan.hpp
        ${CURRENT_COMMON_SOURCE_DIR}/max/max_impl.hpp
        ${CURRENT_COMMON_SOURCE_DIR}/max/block_reduce_max.hpp
        ${CURRENT_COMMON_SOURCE_DIR}/timer/timer.hpp
        ${CURRENT_COMMON_SOURCE_DIR}/tools/set_val.hpp
        )
