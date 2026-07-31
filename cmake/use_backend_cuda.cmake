enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
if (CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION}")
    # message: ${CUDAToolkit_INCLUDE_DIRS}, ${CUDAToolkit_LIBRARY_DIR}
else()
    message(FATAL_ERROR "CUDA Toolkit not found!")
endif()

include_directories(${PROJECT_SOURCE_DIR}/library/uptk)

set(CMAKE_CUDA_FLAGS "${HIP_NVCC_FLAGS} ${HIP_HIPCC_FLAGS}")

if(NOT DEFINED USPARSE_GPU_BACKEND_SPARSE_LIB OR "${USPARSE_GPU_BACKEND_SPARSE_LIB}" STREQUAL "")
    set(USPARSE_GPU_BACKEND_SPARSE_LIB "cusparse")
endif()


function(set_source_files_properties_hip)
    set(SOURCES ${ARGN})
    set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CUDA)
endfunction()

function(hip_add_library_for_cuda target_name)
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

    add_library(${target_name} ${lib_type} ${source_files})

    if(ARG_HIPCC_OPTIONS)
        target_compile_options(${target_name} PRIVATE ${ARG_HIPCC_OPTIONS})
    endif()
endfunction()

# hip_add_library
macro(hip_add_library name)
    hip_add_library_for_cuda(${name} ${ARGN})
endmacro()
