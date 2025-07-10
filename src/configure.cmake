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
endif ()

if (DEVICE_MALLOC_INITILIZE_FLAG)
    set(DEVICE_MALLOC_INITILIZE ON)
    MESSAGE(STATUS "device side malloc initialization")
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
