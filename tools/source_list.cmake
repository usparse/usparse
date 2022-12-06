
set(TOOLS_HEADER_FILES ${TOOLS_HEADER_FILES}
        ${CMAKE_CURRENT_SOURCE_DIR}/common_function.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/matrix_market_reader.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sparse_format.h
        ${CMAKE_CURRENT_SOURCE_DIR}/sort_omp.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/types.h
)

include_directories(
        ${TOOLS_HEADER_FILES}
)
