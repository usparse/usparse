# 测试仅依赖 AMG 本身和 C++ 标准库，不依赖 GoogleTest 或 HIP。
add_executable(amg_cpu_test amg_cpu_test.cpp)
target_link_libraries(amg_cpu_test PRIVATE usparse::amg)
if(MSVC)
    target_compile_options(amg_cpu_test PRIVATE /utf-8)
endif()
set_target_properties(amg_cpu_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    CXX_EXTENSIONS OFF
)

# 单独注册各个规模，失败时 CTest 会指出具体是哪一个例子。
foreach(size IN ITEMS 1 4 9 31 127 511 4096)
    add_test(NAME amg_poisson_${size} COMMAND amg_cpu_test ${size})
    set_tests_properties(amg_poisson_${size} PROPERTIES LABELS "amg;cpu" TIMEOUT 60)
endforeach()

foreach(side IN ITEMS 4 8 16 32)
    add_test(NAME amg_poisson2d_${side} COMMAND amg_cpu_test --poisson2d ${side})
    set_tests_properties(amg_poisson2d_${side} PROPERTIES LABELS "amg;cpu" TIMEOUT 60)
endforeach()

add_test(NAME amg_hierarchy COMMAND amg_cpu_test --hierarchy-test)
set_tests_properties(amg_hierarchy PROPERTIES LABELS "amg;cpu" TIMEOUT 60)

add_test(NAME amg_regression COMMAND amg_cpu_test --self-test)
set_tests_properties(amg_regression PROPERTIES LABELS "amg;cpu" TIMEOUT 60)

# 非法命令行参数应返回非零状态，避免误把失败的求解当作成功。
add_test(NAME amg_invalid_size COMMAND amg_cpu_test 0)
set_tests_properties(amg_invalid_size PROPERTIES LABELS "amg;cpu" WILL_FAIL TRUE TIMEOUT 30)
