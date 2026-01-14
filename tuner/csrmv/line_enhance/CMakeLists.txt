set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/utuner/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/utuner/lib)

add_executable(
        usparse_tuner_csrmv_line_enhance
        line_enhance_tuner.cc
        main.cc
        )
target_include_directories(
        usparse_tuner_csrmv_line_enhance
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}$<SEMICOLON>${CMAKE_SOURCE_DIR}/benchmarks>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(
        usparse_tuner_csrmv_line_enhance
        PRIVATE
        ${USPARSE_TUNER_LIB_NAME}
        ${USPARSE_LIB_NAME}
        ${SPMV_ACC_CLI_LIB}
        )
set_target_properties(usparse_tuner_csrmv_line_enhance PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
)
