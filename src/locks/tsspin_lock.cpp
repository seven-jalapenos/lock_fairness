
#include <algorithm> // min()

#include "tsspin_lock.hpp"
#include <atomic>
#include <cstdint>
#include <immintrin.h> // for _mm_pause (x86)

void TSSpinLock::lock() noexcept {
    uint64_t local_ts = __rdtsc();
    int spins = 1;

    for (;;) {
        // check if local timestamp is older than lock timestamp
        uint64_t lock_ts = next.load(std::memory_order_acquire);
        if (lock_ts > local_ts) {
            spins = 1;
            if (next.compare_exchange_weak(lock_ts, local_ts,
                                           std::memory_order_release, std::memory_order_relaxed)) {
                continue;
            }
        } else if (lock_ts == local_ts) { // we are oldest request
            if (!flag.load(std::memory_order_relaxed)) {
                if (!flag.exchange(true, std::memory_order_acquire)) {
                    // signal that next is no longer taken
                    next.store(UINT64_MAX, std::memory_order_release);
                    return; // got lock (^-^)
                }
                // if exchange fails then we were outran by an older thread
                // reset spins because we are probably close to the oldest
                spins = 1;
            }
        }
        // distribute contention
        for (int i = 0; i < spins; ++i) _mm_pause();
        spins = std::min(spins * 2, 1024);
    }
}

inline void TSSpinLock::unlock() noexcept {
    flag.store(false, std::memory_order_release);
}
