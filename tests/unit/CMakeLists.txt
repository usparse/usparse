add_subdirectory(${PROJECT_SOURCE_DIR}/third-party/googletest-1.16/ gtest-1.16)

enable_testing()
find_package(rocSPARSE REQUIRED)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

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
        roc::rocsparse
        ${USPARSE_KERNEL_LIB_NAME}        
)

include(GoogleTest)
gtest_discover_tests(usparse_test)
