if (HIP_ENABLE_FLAG)
    # rocm
    set(ARCH_NAME hip)
    MESSAGE(STATUS "HIP acceleration is enabled")
else ()
    # cuda
endif ()

if (DEVICE_SIDE_VERIFY_FLAG)
    set(DEVICE_SIDE_VERIFY ON)
    MESSAGE(STATUS "verify in device side")
#     set(gpu ON)
endif ()

# sort kernel strategy
string(TOLOWER ${SORT_KERNEL_STRATEGY} SORT_KERNEL_STRATEGY_LOWER)
if (SORT_KERNEL_STRATEGY_LOWER MATCHES "default")
    set(SORT_KERNEL_STRATEGY_DEFAULT ON)
    set(SORT_KERNEL_STRATEGY_BLOCK_RADIX ON)
elseif (SORT_KERNEL_STRATEGY_LOWER MATCHES "block-radix")
    set(SORT_KERNEL_STRATEGY_BLOCK_RADIX ON)
endif ()

set(__WF_SIZE__ ${WAVEFRONT_SIZE})

configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/building_config.h.in"
        "${PROJECT_BINARY_DIR}/generated/building_config.h"
)

# install the generated file
install(FILES "${PROJECT_BINARY_DIR}/generated/building_config.h"
        DESTINATION "include"
        )
