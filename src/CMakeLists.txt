# set binary path and lib storage path for compiled files.
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

add_executable(
    gmres-host
    main.cpp 
    gmres.cpp
    )

target_include_directories(gmres-host PUBLIC ${PROJECT_SOURCE_DIR}/inc)
