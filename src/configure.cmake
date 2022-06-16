if (HIP_ENABLE_FLAG)
    # rocm
    set(ARCH_NAME hip)
    MESSAGE(STATUS "HIP acceleration is enabled")
else ()
    # cuda
endif ()

if (DEVICE_SIDE_VERIFY_FLAG)
    set(DEVICE_SIDE_VERIFY ON)
    set(gpu ON)
endif ()

string(TOLOWER ${KERNEL_STRATEGY} KERNEL_STRATEGY_LOWER)
if (KERNEL_STRATEGY_LOWER MATCHES "default")
    set(KERNEL_STRATEGY_DEFAULT ON)
elseif (KERNEL_STRATEGY_LOWER MATCHES "kokkos")
    set(KERNEL_STRATEGY_KOKKOS ON)
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
