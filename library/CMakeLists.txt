include(configure.cmake) # spmv-acc

set(USPARSE_HEADER_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include/)
set(USPARSE_API_PATH ${CMAKE_CURRENT_SOURCE_DIR}/api/)
set(USPARSE_HIP_SOURCE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/hip/)

if (USPARSE_ENABLE_HIP)
    add_subdirectory(hip)
endif()

## gmres-gpu start
# set binary path and lib storage path for compiled files.
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

set(GMRES_HEADER_FILES
    dtype.h
    gmres_crs.hpp
    read_crs.hpp
    )

set(GMRES_SOURCE_FILES
    GMRES.cpp
    )


set(GMRES_LIB_NAME "gmres")
add_library(${GMRES_LIB_NAME} STATIC ${GMRES_HEADER_FILES} ${GMRES_SOURCE_FILES})

set(EXECUTE_BIN_NAME "gmres-gpu")
add_executable(${EXECUTE_BIN_NAME} main.cpp)

target_link_libraries(${EXECUTE_BIN_NAME}
    PRIVATE ${GMRES_LIB_NAME}
)
## gmres-gpu end
