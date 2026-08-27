
#include <atomic>
#include <cstdint>
#include <cstdlib> // abort()

#include <immintrin.h> // _mm_pause
#include <x86intrin.h> // __rdtsc

#include "tssel_lock.hpp"

// Slot ids persist for the thread's lifetime. Like the other locks here this
// assumes one lock instance per run, which is all the benchmark ever builds.
static thread_local int tssel_local_id = -1;

int TSSelectLock::local_id() noexcept {
    if (tssel_local_id == -1) {
        tssel_local_id = registered.fetch_add(1, std::memory_order_acq_rel);
        // Wrapping the id instead would silently give two threads the same slot,
        // which breaks selection rather than just degrading it.
        if (tssel_local_id >= MAX_THREADS) std::abort();
    }
    return tssel_local_id;
}

// Withdraw our request and try to grab a lock that looks free. The withdrawal
// CAS is what keeps this safe: whoever wins it -- us, or a scanner about to
// grant us the lock -- owns the request, so a thread can never both be granted
// and self-acquire.
bool TSSelectLock::try_take_free(int me, uint64_t my_ts) noexcept {
    uint64_t expected = my_ts;
    if (!slots[me].ts.compare_exchange_strong(expected, NO_REQUEST,
                                              std::memory_order_acq_rel,
                                              std::memory_order_relaxed)) {
        return false; // a scanner claimed us; the grant is already on its way
    }

    bool expected_free = false;
    if (held.compare_exchange_strong(expected_free, true,
                                     std::memory_order_acquire,
                                     std::memory_order_relaxed)) {
        return true;
    }

    // Lost the race. Re-publish the *original* timestamp rather than a fresh
    // one so a failed attempt doesn't demote us to the back of the age order.
    slots[me].ts.store(my_ts, std::memory_order_release);
    return false;
}

// Find the oldest pending request and claim it. Only the current owner ever
// runs this, so the scan itself needs no mutual exclusion; the claim still has
// to be a CAS because a waiter may withdraw concurrently via try_take_free.
int TSSelectLock::select_oldest() noexcept {
    for (;;) {
        const int n = registered.load(std::memory_order_relaxed);
        uint64_t best_ts = NO_REQUEST;
        int best = NO_SUCC;

        for (int i = 0; i < n; ++i) {
            uint64_t ts = slots[i].ts.load(std::memory_order_acquire);
            if (ts < best_ts) {
                best_ts = ts;
                best = i;
            }
        }
        if (best == NO_SUCC) return NO_SUCC;

        uint64_t expected = best_ts;
        if (slots[best].ts.compare_exchange_strong(expected, NO_REQUEST,
                                                   std::memory_order_acq_rel,
                                                   std::memory_order_relaxed)) {
            return best;
        }
        // The thread we picked withdrew to grab a momentarily free lock; the
        // remaining candidates may have shifted, so start the scan over.
    }
}

void TSSelectLock::lock() noexcept {
    // Read the clock before anything else so the published age reflects when we
    // asked for the lock, not how long we spent losing the fast-path CAS.
    const uint64_t my_ts = __rdtsc();

    // Uncontended path: take a free lock without ever publishing a request.
    if (!held.load(std::memory_order_relaxed)) {
        bool expected_free = false;
        if (held.compare_exchange_strong(expected_free, true,
                                         std::memory_order_acquire,
                                         std::memory_order_relaxed)) {
            return;
        }
    }

    const int me = local_id();
    slots[me].go.store(false, std::memory_order_relaxed);
    slots[me].ts.store(my_ts, std::memory_order_release);

    for (;;) {
        if (slots[me].go.load(std::memory_order_acquire)) {
            slots[me].go.store(false, std::memory_order_relaxed);
            return; // handed the lock directly by the previous owner
        }
        _mm_pause();

        // Safety net for a request published just after the last scan read our
        // slot. `held` stays true across handoffs, so this only touches the
        // shared line on the rare occasions the lock actually falls idle.
        if (!held.load(std::memory_order_relaxed) && try_take_free(me, my_ts)) {
            return;
        }
    }
}

void TSSelectLock::unlock() noexcept {
    // Consume the successor the previous owner picked for us, marking the slot
    // SCAN_PENDING to claim the scanner role. Reading SCAN_PENDING means that
    // owner is still scanning, which only costs us when our critical section
    // was shorter than a scan -- otherwise selection was free.
    int succ;
    for (;;) {
        succ = next_id.load(std::memory_order_acquire);
        if (succ == SCAN_PENDING) {
            _mm_pause();
            continue;
        }
        if (next_id.compare_exchange_weak(succ, SCAN_PENDING,
                                          std::memory_order_acq_rel,
                                          std::memory_order_relaxed)) {
            break;
        }
    }

    if (succ == NO_SUCC) {
        // Nobody was waiting as of the last scan. Look once more for a request
        // that arrived since, then drop the lock if there genuinely is none.
        succ = select_oldest();
        if (succ == NO_SUCC) {
            // Clear the scan marker before releasing: the next owner arrives by
            // CAS rather than by grant, and would otherwise spin on a stale
            // SCAN_PENDING that no one is ever going to publish over.
            next_id.store(NO_SUCC, std::memory_order_relaxed);
            held.store(false, std::memory_order_release);
            return;
        }
    }

    slots[succ].go.store(true, std::memory_order_release);

    // Off the handoff path: choose the successor for the thread we just granted.
    next_id.store(select_oldest(), std::memory_order_release);
}
