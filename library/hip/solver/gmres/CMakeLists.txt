# set binary path and lib storage path for compiled files.
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

# set binary path and lib storage path for compiled files.
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

# build static lib
set(USPARSE_GMRES_HEADER_FILES ${USPARSE_GMRES_HEADER_FILES}
        kernels/usparse_gmres.h
        kernels/level1/usparse_level1_batched.h
        kernels/level2/usparse_csrmv.h
        kernels/level2/usparse_rotation.h
        kernels/level2/usparse_solver_tri.h

)
set(USPARSE_GMRES_SOURCE_FILES ${USPARSE_GMRES_SOURCE_FILES}
        kernels/level1/usparse_dotprod_axpy_fusion.cpp
        kernels/level1/usparse_level1_batched.cpp
        kernels/level2/usparse_csrmv.cpp
        kernels/level2/usparse_rotation.cpp
        kernels/level2/usparse_solver_tri.cpp
)

set_source_files_properties_hip(${USPARSE_GMRES_SOURCE_FILES})
hip_add_library(${USPARSE_GMRES_KERNEL_LIB_NAME} STATIC ${USPARSE_GMRES_SOURCE_FILES} ${USPARSE_GMRES_HEADER_FILES}) # HIPCC_OPTIONS ${USPARSE_HIPCC_OPTIONS})

message(STATUS ${USPARSE_GMRES_KERNEL_LIB_NAME})

target_include_directories(
        ${USPARSE_GMRES_KERNEL_LIB_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}$<SEMICOLON>${USPARSE_HEADER_PATH}/$<SEMICOLON>${USPARSE_HEADER_PATH}/../$<SEMICOLON>${USPARSE_HEADER_PATH}/../hip/$<SEMICOLON>${PROJECT_BINARY_DIR}/generated>
        $<INSTALL_INTERFACE:include>
)

find_package(rocSPARSE REQUIRED)

add_library(
    gmres-device
    precision_patch.cpp
    gmres.cpp
    )

target_link_libraries(gmres-device PRIVATE
    roc::rocsparse
    hipblas
    hipsparse
    ${USPARSE_GMRES_KERNEL_LIB_NAME}
    )


##########################
##### install files
##########################
# install bin and libs
install(TARGETS gmres-device ${USPARSE_GMRES_KERNEL_LIB_NAME}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        )

# install header.
install(DIRECTORY "./"
        DESTINATION "include"
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp"
        PATTERN "*.inl"
        )
