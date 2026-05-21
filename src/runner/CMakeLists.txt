
# Main benchmarking and logging executable
add_executable(lock_exe 
    main.cpp 
    logging.cpp
    rdtsc_offsets.cpp
)

# Link both to our locks library and thread support
target_link_libraries(lock_exe PRIVATE locks Threads::Threads)

# Allow the test folder to find its own headers (like pin_thread.hpp)
target_include_directories(lock_exe PRIVATE
    ${CMAKE_SOURCE_DIR}/include/runner
)
target_include_directories(lock_exe PRIVATE
    ${CMAKE_SOURCE_DIR}/include/locks
)

# Set output directory to the project's bin folder
set_target_properties(lock_exe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/files/logs")