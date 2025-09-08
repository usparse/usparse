option(HIP_ENABLE_FLAG "Enable HIP" ON)

option(USPARSE_UNIT_TEST_ENABLE "Enable unit tests" ON)
option(USPARSE_BENCHMARK_ENABLE "Enable benchmark" ON)

option(GMRES_TOOLS_ENABLE "Enable tools" ON)
option(USPARSE_AUTO_TUNER_ENABLE "Enable tools" ON) # it depends on GMRES_TOOLS_ENABLE

set(USPARSE_KERNEL_LIB_NAME "usparse_kernels")
set(USPARSE_TUNER_LIB_NAME "usparse_tuning")
