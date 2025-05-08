option(USPARSE_ENABLE_HIP "Enable HIP" ON)
option(UPSARSE_BUILD_BENCHMARK "build benchmark" OFF)
option(UPSARSE_DEVICE_SIDE_VERIFY "Verify result of device side" OFF)
option(UPSARSE_BUILD_FOR_ROCSPARSE "build lib for rocsparse" OFF)
set(UPSARSE_WAVEFRONT_SIZE "64" CACHE STRING "threads number in a wavefront")

set(USPARSE_LIB_NAME usparse)
