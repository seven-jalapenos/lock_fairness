
#pragma once
#include <cstdint>
#include <x86intrin.h> // For rdtscp intrinsic

static inline uint64_t rdtscp(uint32_t &cpu_aux) {
    return __rdtscp(&cpu_aux);
}
