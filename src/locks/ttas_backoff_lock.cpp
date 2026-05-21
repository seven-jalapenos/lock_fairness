
#include "ttas_backoff_lock.hpp"
#include <algorithm>
#include <immintrin.h> // for _mm_pause (x86)

void TTASLock_Backoff::lock() noexcept {
   for (;;) {
       // First phase: spin-read (shared loads, no invalidation)
       while (flag.load(std::memory_order_relaxed)) {
           int spins = 1;
           while (flag.load(std::memory_order_relaxed)) {
               for (int i = 0; i < spins; ++i) _mm_pause();
               spins = std::min(spins * 2, 1024);
           }
       }
       // Second phase: attempt to acquire (RMW)
       if (!flag.exchange(true, std::memory_order_acquire)) {
           return; // acquired
       }
       // Failed: someone else beat us, retry
   }
}

inline void TTASLock_Backoff::unlock() noexcept {
    flag.store(false, std::memory_order_release);
}
