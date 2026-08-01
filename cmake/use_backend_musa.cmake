list(APPEND CMAKE_MODULE_PATH /usr/local/musa/cmake)
find_package(MUSA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/library/uptk)

set(CMAKE_CUDA_FLAGS "${HIP_NVCC_FLAGS} ${HIP_HIPCC_FLAGS}")
set(USPARSE_LINK_LINBS "musart") # cusparse, musparse

if(NOT DEFINED USPARSE_GPU_BACKEND_SPARSE_LIB OR "${USPARSE_GPU_BACKEND_SPARSE_LIB}" STREQUAL "")
    set(USPARSE_GPU_BACKEND_SPARSE_LIB "musparse")
endif()

function(hip_add_library_for_musa target_name)
    set(options STATIC SHARED)
    set(oneValueArgs)
    set(multiValueArgs HIPCC_OPTIONS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(ARG_STATIC)
        set(lib_type STATIC)
    elseif(ARG_SHARED)
        set(lib_type SHARED)
    else()
        message(FATAL_ERROR "hip_add_library: can only be STATIC or SHARED")
    endif()

    set(source_files ${ARG_UNPARSED_ARGUMENTS})
    
    if(ARG_HIPCC_OPTIONS)
        set(MUSA_MCC_FLAGS ${ARG_HIPCC_OPTIONS})
    endif()
    set(MUSA_LINK_LIBRARIES_KEYWORD INTERFACE)
    musa_add_library(${target_name} ${lib_type} ${source_files})
endfunction()

# hip_add_library
macro(hip_add_library name)
    hip_add_library_for_musa(${name} ${ARGN})
endmacro()

function(set_source_files_properties_hip)
    set(SOURCES ${ARGN})
    set_source_files_properties(${SOURCES} PROPERTIES MUSA_SOURCE_PROPERTY_FORMAT OBJ)
endfunction()
