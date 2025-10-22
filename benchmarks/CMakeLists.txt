set(USPARSE_BENCHMARK_UTILS_LIB benchmarkutil)
add_subdirectory(utils)

add_subdirectory(coomv) # the coomv benchmark
add_subdirectory(csrmv) # the legency spmv-acc benchmark
add_subdirectory(csr_spgemm) # the legency spgemm benchmark
add_subdirectory(coomm) # the spmm benchmark for coo format
add_subdirectory(csrmm) # the spmm benchmark for csr format
include(csrsm.cmake) # todo: move to dir csrsm
