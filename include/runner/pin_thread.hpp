
#pragma once
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>


inline void pin_thread_to_core(int core_id) {
    int num_cores = sysconf(_SC_NPROCESSORS_ONLN);

    if (core_id < 0 || core_id >= num_cores) {
        fprintf(stderr, "Invalid core_id %d (max %d)\n", core_id, num_cores - 1);
        std::abort();
    }

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);

    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
        perror("pthread_setaffinity_np");
        std::abort();
    }
}