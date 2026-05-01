add_subdirectory(${PROJECT_SOURCE_DIR}/third-party/googletest-1.16/ gtest-1.16)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        if(TARGET gtest)
                target_compile_options(gtest PRIVATE -Wno-return-type)
        endif()

        if(TARGET gtest_main)
                target_compile_options(gtest_main PRIVATE -Wno-return-type)
        endif()

        if(TARGET gmock)
                target_compile_options(gmock PRIVATE -Wno-return-type)
        endif()

        if(TARGET gmock_main)
                target_compile_options(gmock_main PRIVATE -Wno-return-type)
        endif()
endif()

if (BENCHMARK_CUDA_ENABLE_FLAG)
    set(BENCHMARK_DEP_LIBS
        ${BENCHMARK_DEP_LIBS}
        cusparse
        )
else ()
    set(BENCHMARK_DEP_LIBS
        ${BENCHMARK_DEP_LIBS}
        rocsparse
        )
endif ()

enable_testing()

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

set_source_files_properties_hip(
        usparse/usparse_level1.cc
        usparse/usparse_level1_dotprod_axpy_fusion.cc
        usparse/usparse_level2_csrmv.cc
)
add_executable(
        usparse_test
        usparse/usparse_level1.cc
        usparse/usparse_level1_dotprod_axpy_fusion.cc
        usparse/usparse_level2_csrmv.cc
)
target_include_directories(
        usparse_test
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>        
)

target_link_libraries(
        usparse_test
        GTest::gtest_main
        ${BENCHMARK_DEP_LIBS}
        ${USPARSE_GMRES_KERNEL_LIB_NAME}
        ${USPARSE_LIB_NAME}
)

include(GoogleTest)
gtest_discover_tests(usparse_test)
