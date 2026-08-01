set_source_files_properties_hip(mtx_analysis_lib.cpp)

add_library(mtx_analysis_lib mtx_analysis_lib.cpp)
target_include_directories(
        mtx_analysis_lib
        PUBLIC
        ${PROJECT_SOURCE_DIR}/benchmarks/solver/gmres
)
target_link_libraries(
        mtx_analysis_lib
        PUBLIC
        ${USPARSE_GMRES_KERNEL_LIB_NAME}
        ${USPARSE_LIB_NAME}
)

add_executable(
    mtx_analysis
    main.cpp
    )

target_link_libraries(mtx_analysis PRIVATE mtx_analysis_lib)

target_link_options(mtx_analysis PUBLIC ${HIP_NVCC_LINKER_FLAGS})
