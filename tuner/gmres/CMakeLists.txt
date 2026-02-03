find_package(rocSPARSE REQUIRED)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/utuner/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/utuner/lib)

add_executable(
        usparse_kernel_auto_tuner
        fusion/dotprod_axpy_fusion_kernel_tuner.cc
        main.cc
)

target_include_directories(
        usparse_kernel_auto_tuner
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>        
)
set_target_properties(usparse_kernel_auto_tuner PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
)

target_link_libraries(
        usparse_kernel_auto_tuner
        roc::rocsparse
        ${USPARSE_GMRES_KERNEL_LIB_NAME}
        ${USPARSE_TUNER_LIB_NAME}
)
