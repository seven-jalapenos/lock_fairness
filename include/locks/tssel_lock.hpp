
#pragma once
#include <atomic>
#include <cstdint>

#include "lockADT.hpp"

// Timestamp-selection lock. Waiters publish an rdtsc timestamp into a shared
// slot array; the outgoing owner hands the lock directly to the oldest pending
// request and then scans for the one after that, so the O(n) selection overlaps
// the next owner's critical section instead of sitting in the handoff path.
// The point of the design is that grant order follows *invocation* order rather
// than the order threads happened to win an atomic RMW, which is what the queue
// locks (MCS/CLH/ticket) actually order by.
class TSSelectLock : public Lock {
private:
    // Hard cap on distinct threads that may ever touch this lock; slots are
    // handed out once per thread and never recycled.
    static constexpr int MAX_THREADS = 128;

    static constexpr uint64_t NO_REQUEST = UINT64_MAX; // slot holds no pending request
    static constexpr int NO_SUCC = -1;                 // scan found nobody waiting
    static constexpr int SCAN_PENDING = -2;            // previous owner's scan still running

    // One cache line per thread. Scanners only *read* `ts`, and a waiter only
    // reads its own `go`, so a spinning waiter and a scanning owner share the
    // line without ping-ponging it; the single write comes when the owner
    // claims the slot and grants it, which is one line transfer for both fields.
    struct alignas(64) Slot {
        std::atomic<uint64_t> ts{NO_REQUEST};
        std::atomic<bool> go{false};
    };

    Slot slots[MAX_THREADS];

    // Stays true across a direct handoff, so the lock is never observably free
    // between two owners and an arriving thread cannot cut the line.
    alignas(64) std::atomic<bool> held{false};
    // Successor picked by the previous owner, or one of the sentinels above.
    alignas(64) std::atomic<int> next_id{NO_SUCC};
    // Doubles as the slot-id allocator and the scan bound.
    alignas(64) std::atomic<int> registered{0};

    int local_id() noexcept;
    bool try_take_free(int me, uint64_t my_ts) noexcept;
    int select_oldest() noexcept;

public:
    TSSelectLock() = default;
    ~TSSelectLock() override = default;

    void lock() noexcept override;
    void unlock() noexcept override;
};
