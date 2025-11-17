set(CURRENT_COMMON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(COMMON_HEADER_FILES ${COMMON_HEADER_FILES}
        ${CURRENT_COMMON_SOURCE_DIR}/common.h
        ${CURRENT_COMMON_SOURCE_DIR}/CPU/hip_check.hpp
        ${CURRENT_COMMON_SOURCE_DIR}/DCU/scan/scan_impl.hpp
        ${CURRENT_COMMON_SOURCE_DIR}/DCU/scan/scan_in_device.hpp
        ${CURRENT_COMMON_SOURCE_DIR}/DCU/scan/reduce_then_scan.hpp
        ${CURRENT_COMMON_SOURCE_DIR}/DCU/scan/lookback_scan.hpp
        ${CURRENT_COMMON_SOURCE_DIR}/DCU/scan/single_block_scan.hpp
        ${CURRENT_COMMON_SOURCE_DIR}/DCU/max/max_impl.hpp
        ${CURRENT_COMMON_SOURCE_DIR}/DCU/max/block_reduce_max.hpp
        ${CURRENT_COMMON_SOURCE_DIR}/CPU/timer.h
        ${CURRENT_COMMON_SOURCE_DIR}/DCU/utils/set_val.hpp
        )
