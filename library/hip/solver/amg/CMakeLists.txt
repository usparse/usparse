add_library(usparse_amg STATIC amg.cpp amg.hpp amg_types.hpp)
add_library(usparse::amg ALIAS usparse_amg)

target_compile_features(usparse_amg PUBLIC cxx_std_14)
if(MSVC)
    target_compile_options(usparse_amg PRIVATE /utf-8)
endif()
target_include_directories(usparse_amg PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
set_target_properties(usparse_amg PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    CXX_EXTENSIONS OFF
)
