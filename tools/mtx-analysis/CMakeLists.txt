add_executable(
    mtx_analysis
    main.cpp
    )

target_include_directories(mtx_analysis PUBLIC ${PROJECT_SOURCE_DIR}/benchmarks/solver/gmres)
target_link_libraries(mtx_analysis PRIVATE
#         roc::rocsparse
        ${USPARSE_GMRES_KERNEL_LIB_NAME}
        )
