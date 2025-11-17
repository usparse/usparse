add_executable(
    mtx_analysis
    main.cpp
    )

target_include_directories(mtx_analysis PUBLIC ${PROJECT_SOURCE_DIR}/inc)
target_link_libraries(mtx_analysis PRIVATE
#         roc::rocsparse
        ${USPARSE_KERNEL_LIB_NAME}
        )
