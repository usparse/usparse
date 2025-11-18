set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

# build static lib
set(USPARSE_TUNER_HEADER_FILES ${USPARSE_TUNER_HEADER_FILES}
        usparse_perf_record.h 
        usparse_tuner.h 
        usparse_tuner_building_config.h
        usparse_tuner_config.hpp
        usparse_tuner_multi_levels.hpp
        usparse_tuner_states.hpp
        usparse_tuner_utils.hpp
)

set(USPARSE_TUNER_SOURCE_FILES ${USPARSE_TUNER_SOURCE_FILES}
        usparse_perf_record.cc
        usparse_tuner.cc
)

add_library(${USPARSE_TUNER_LIB_NAME} STATIC ${USPARSE_TUNER_SOURCE_FILES} ${USPARSE_TUNER_HEADER_FILES})
target_include_directories(
        ${USPARSE_TUNER_LIB_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

set_target_properties(${USPARSE_TUNER_LIB_NAME} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
)

add_executable(
        usparse_tuner_example_sqrt
        example_main.cc
        )
target_link_libraries(
        usparse_tuner_example_sqrt
        PRIVATE
        ${USPARSE_TUNER_LIB_NAME}
        )
set_target_properties(usparse_tuner_example_sqrt PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
)
