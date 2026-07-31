# see https://github.com/ROCm-Developer-Tools/HIP/blob/master/samples/2_Cookbook/12_cmake_hip_add_executable
if (NOT DEFINED HIP_PATH)
    if (NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else ()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif ()
endif ()
set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})

find_package(HIP REQUIRED)
if (HIP_FOUND)
    message(STATUS "Found HIP: " ${HIP_VERSION})
else ()
    message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location.")
endif ()

set(USPARSE_HIPCC_OPTIONS) # set hipcc arguments

if(NOT DEFINED USPARSE_GPU_BACKEND_SPARSE_LIB OR "${USPARSE_GPU_BACKEND_SPARSE_LIB}" STREQUAL "")
    set(USPARSE_GPU_BACKEND_SPARSE_LIB "rocsparse")
endif()

function(set_source_files_properties_hip)
    set(SOURCES ${ARGN})
    # for HIP language:
    # set_source_files_properties(${SOURCES} PROPERTIES
    #     LANGUAGE HIP
    #     HIP_SOURCE_PROPERTY_FORMAT 1
    # )
    set_source_files_properties(${SOURCES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
endfunction()
