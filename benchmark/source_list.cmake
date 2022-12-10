
set(BENCKMARK_HEADER_FILES ${BENCHMARK_HEADER_FILES}
        ${CMAKE_CURRENT_SOURCE_DIR}/cli/sort_omp.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cli/martix_market_reader.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cli/sparse_format.h
        ${CMAKE_CURRENT_SOURCE_DIR}/cli/types.h
        ${CMAKE_CURRENT_SOURCE_DIR}/common.h
        # ${CMAKE_CURRENT_SOURCE_DIR}/memory_copy.hpp
        # ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_csr2csc.hpp
)

include_directories(
        ${BENCHMARK_HEADER_FILES}
)
