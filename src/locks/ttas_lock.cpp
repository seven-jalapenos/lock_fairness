
#include "ttas_lock.hpp"
#include <immintrin.h> // for _mm_pause (x86)


void TTASLock::lock() noexcept {
    for (;;) {
        // First phase: spin-read (shared loads, no invalidation)
        while (flag.load(std::memory_order_relaxed)) {
            _mm_pause(); // reduce contention / power
        }
        // Second phase: attempt to acquire (RMW)
        if (!flag.exchange(true, std::memory_order_acquire)) {
            return; // acquired
        }
        // Failed: someone else beat us, retry
    }
}

inline void TTASLock::unlock() noexcept {
    flag.store(false, std::memory_order_release);
}
