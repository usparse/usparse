include(configure.cmake) # spmv-acc

set(USPARSE_HEADER_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include/)
set(USPARSE_API_PATH ${CMAKE_CURRENT_SOURCE_DIR}/api/)
set(USPARSE_HIP_SOURCE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/hip/)

add_subdirectory(hip/solver/amg)

if (USPARSE_ENABLE_HIP)
    add_subdirectory(hip)
endif()
