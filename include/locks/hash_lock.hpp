
#pragma once
#include <atomic>
#include <cstdint>

#include "lockADT.hpp"
#include "dumb_hash_map.hpp"

// A ticket lock whose "now serving" signalling is spread across an array of
// per-slot spin locations (a DumbHashMap) instead of a single shared counter,
// à la Anderson's array lock. lock() fetch-and-adds a global ticket and spins
// on the slot the ticket hashes to; unlock() opens the slot that ticket + 1
// hashes to, waking exactly the next ticket holder.
class HashLock : public Lock {
private:
    alignas(64) std::atomic<uint64_t> next_ticket{0};
    // Ticket of the thread currently holding the lock. Only touched by the one
    // thread inside the critical section (write on acquire, read on release),
    // so it needs no atomicity of its own. Kept on its own cache line to avoid
    // false sharing with the hammered next_ticket counter.
    alignas(64) uint64_t held_ticket{0};
    DumbHashMap map;

public:
    HashLock();
    ~HashLock() override = default;

    void lock() noexcept override;
    void unlock() noexcept override;
};
