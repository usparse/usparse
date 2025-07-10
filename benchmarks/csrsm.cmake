set(USPARSE_CSRSM_SOURCE_FILES
        csrsm/common.h
        csrsm/matrix_utils.cpp
        csrsm/matrix_utils.h
        csrsm/rocsparse_test.cpp
        csrsm/rocsparse_test.h
        csrsm/updown_test.cpp
        csrsm/verification.cpp
        csrsm/verification.h
        csrsm.cpp
    )

add_executable(csrsm-gpu-benchmark ${USPARSE_CSRSM_SOURCE_FILES})
target_link_libraries(csrsm-gpu-benchmark
        PUBLIC
        ${USPARSE_LIB_NAME}
        ${SPMV_ACC_CLI_LIB}
        rocsparse # link rocsparse
)
