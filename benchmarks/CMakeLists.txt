set(USPARSE_BENCHMARK_UTILS_LIB benchmarkutil)
add_subdirectory(utils)

add_subdirectory(csrmv) # the legency spmv-acc benchmark
add_subdirectory(csr_spgemm) # the legency spgemm benchmark
include(csrsm.cmake) # todo: move to dir csrsm
