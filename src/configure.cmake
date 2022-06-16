if (DEVICE_SIDE_VERIFY_FLAG)
    set(DEVICE_SIDE_VERIFY ON)
#     set(gpu ON)
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
