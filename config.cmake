option(HIP_ENABLE_FLAG "Enable HIP" ON)
option(DEVICE_SIDE_VERIFY_FLAG "Verify result of device side" OFF)

if(DEVICE_SIDE_VERIFY_FLAG)
    add_definitions(-Dgpu)
endif()

set(WAVEFRONT_SIZE "64" CACHE STRING "threads number in a wavefront")
set(KERNEL_STRATEGY "DEFAULT" CACHE STRING "SpGEMM strategy")
set(SORT_KERNEL_STRATEGY "DEFAULT" CACHE STRING "Sort strategy")

set(SPGEMM_BIN_NAME spgemm-hip)
set(SPGEMM_KERNEL_LIB_NAME spgemm-kernels)
set(SORT_KERNEL_LIB_NAME sort-kernels)
set(ACC_LIBS ${ACC_LIBS} ${SPGEMM_KERNEL_LIB_NAME} ${SORT_KERNEL_LIB_NAME})

if (DEVICE_SIDE_VERIFY_FLAG)
    set(ACC_LIBS ${ACC_LIBS} rocsparse)
endif()
