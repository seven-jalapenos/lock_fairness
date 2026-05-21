
#pragma once
#include <cstdint>

static inline uint64_t rdtscp(uint32_t &aux) {
    uint32_t lo, hi;
    asm volatile ("rdtscp" : "=a"(lo), "=d"(hi), "=c"(aux) ::);
    return ((uint64_t)hi << 32) | lo;
}

void find_offsets();