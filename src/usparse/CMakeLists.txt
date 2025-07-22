# set binary path and lib storage path for compiled files.
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

# build static lib
set(USPARSE_HEADER_FILES ${USPARSE_HEADER_FILES}
        usparse_gmres.h
        kernels/level1/usparse_level1.h
        kernels/level2/usparse_csrmv.h
        kernels/level2/usparse_rotation.h
        kernels/level2/usparse_solver_tri.h
        kernels/spmv/api/types.h
        kernels/spmv/api/spmv_flat.h
        kernels/spmv/api/spmv_line_enhance.h
        kernels/spmv/api/spmv_vec_row.h
	kernels/spmv/api/building_config.h

)
set(USPARSE_SOURCE_FILES ${USPARSE_SOURCE_FILES}
        kernels/level1/usparse_level1.cpp
        kernels/level2/usparse_csrmv.cpp
        kernels/level2/usparse_rotation.cpp
        kernels/level2/usparse_solver_tri.cpp
        kernels/spmv/api/spmv_flat_imp.cpp
        kernels/spmv/api/spmv_line_enhance_imp.cpp
        kernels/spmv/api/spmv_vec_row_imp.cpp
)

include(kernels/spmv/hip-flat/source_list.cmake)
include(kernels/spmv/hip-line-enhance/source_list.cmake)
include(kernels/spmv/hip-vector-row/source_list.cmake)
list(APPEND USPARSE_SOURCE_FILES ${ACC_SOURCE_FILES})
list(APPEND USPARSE_HEADER_FILES ${ACC_HEADER_FILES})

set_source_files_properties(${USPARSE_SOURCE_FILES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_library(${USPARSE_KERNEL_LIB_NAME} STATIC ${USPARSE_SOURCE_FILES} ${USPARSE_HEADER_FILES} HIPCC_OPTIONS ${USPARSE_HIPCC_OPTIONS})
target_include_directories(
        ${USPARSE_KERNEL_LIB_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

##########################
##### install files
##########################
# install bin and libs
install(TARGETS ${USPARSE_KERNEL_LIB_NAME}
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
