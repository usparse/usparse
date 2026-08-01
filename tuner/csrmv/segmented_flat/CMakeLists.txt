set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/utuner/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/utuner/lib)

set_source_files_properties_hip(
        segmented_flat_tuner.cc
        ../common/csrmv_tuner_main.cc
        )

add_executable(
        usparse_tuner_csrmv_segmented_flat
        segmented_flat_tuner.cc
        ../common/csrmv_tuner_main.cc
        )
target_include_directories(
        usparse_tuner_csrmv_segmented_flat
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}$<SEMICOLON>${CMAKE_SOURCE_DIR}/benchmarks>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(
        usparse_tuner_csrmv_segmented_flat
        PRIVATE
        ${USPARSE_TUNER_LIB_NAME}
        ${USPARSE_LIB_NAME}
        ${SPMV_ACC_CLI_LIB}
)
set_target_properties(usparse_tuner_csrmv_segmented_flat PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
)
