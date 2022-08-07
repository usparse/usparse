option(HIP_ENABLE_FLAG "Enable HIP" ON)
option(DEVICE_SIDE_VERIFY_FLAG "Verify result of device side" ON)

if(DEVICE_SIDE_VERIFY_FLAG)
    add_definitions(-Dgpu)
endif()

set(WAVEFRONT_SIZE "64" CACHE STRING "threads number in a wavefront")
set(KERNEL_STRATEGY "DEFAULT" CACHE STRING "SpGEMM strategy")

#每个策略编译为一个动态库
#设置各个动态库的名字
set(SPGEMM_BIN_NAME spgemm-dense2csr)
set(SPGEMM_DENSE_KERNEL_LIB_NAME spgemm-dense2csr-kernels)
set(ACC_LIBS ${ACC_LIBS} ${SPGEMM_DENSE_KERNEL_LIB_NAME})

if (DEVICE_SIDE_VERIFY_FLAG)
    set(ACC_LIBS ${ACC_LIBS} rocsparse)
endif()
