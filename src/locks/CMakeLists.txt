
add_library(locks STATIC
    clh_lock.cpp
    mcs_lock.cpp
    ticket_lock.cpp
    ttas_backoff_lock.cpp
    ttas_lock.cpp
)

# Tell CMake where the public headers are
target_include_directories(locks PUBLIC 
    ${CMAKE_SOURCE_DIR}/include/locks
)

# Link threads if necessary for the library itself
target_link_libraries(locks PUBLIC Threads::Threads)