
cmake_minimum_required(VERSION 3.15)
project(LockAnalysis VERSION 1.0 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# Executables (.exe, .out)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
# Static libraries (.lib, .a)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Require C++20 for modern alignment and synchronization primitives
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization is crucial for lock benchmarking
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

# Ensure threads are available (pthread)
find_package(Threads REQUIRED)

# Add subdirectories
add_subdirectory(src)