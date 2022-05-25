option(HIP_ENABLE_FLAG "Enable HIP" ON)

set(WAVEFRONT_SIZE "64" CACHE STRING "threads number in a wavefront")
set(KERNEL_STRATEGY "DEFAULT" CACHE STRING "SpGEMM strategy")

set(SPGEMM_BIN_NAME spgemm-hip)
set(SPGEMM_KERNEL_LIB_NAME spgemm-kernels)
set(ACC_LIBS ${ACC_LIBS} ${SPGEMM_KERNEL_LIB_NAME})
