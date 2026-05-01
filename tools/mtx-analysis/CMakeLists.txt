set_source_files_properties_hip(main.cpp)

add_executable(
    mtx_analysis
    main.cpp
    )

target_include_directories(mtx_analysis PUBLIC ${PROJECT_SOURCE_DIR}/benchmarks/solver/gmres)
target_link_libraries(mtx_analysis PRIVATE
#         roc::rocsparse
        ${USPARSE_GMRES_KERNEL_LIB_NAME}
        )

target_link_options(mtx_analysis PUBLIC ${HIP_NVCC_LINKER_FLAGS})
