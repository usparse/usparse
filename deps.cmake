# find HIP if HIP is enabled, and embeded UPTK is not enabled.
if (USPARSE_ENABLE_HIP AND (NOT USPARSE_USE_EMBEDED_UPTK))
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
endif ()

# set UPTK path: use CUDA env and use uptk to support HIP.
if (USPARSE_ENABLE_HIP AND USPARSE_USE_EMBEDED_UPTK)
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

    # hip_add_library
    macro(hip_add_library name)
        add_library(${name} ${ARGN})
    endmacro()
endif ()

function(set_source_files_properties_hip)
    set(SOURCES ${ARGN})
    if(NOT USPARSE_USE_EMBEDED_UPTK)
        set_source_files_properties(${SOURCES} PROPERTIES
            LANGUAGE HIP
            HIP_SOURCE_PROPERTY_FORMAT 1
        )
    elseif(USPARSE_USE_EMBEDED_UPTK)
        set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CUDA)
    else()
        message(FATAL_ERROR "Unsupported GPU_BACKEND: ${GPU_BACKEND}")
    endif()
endfunction()
