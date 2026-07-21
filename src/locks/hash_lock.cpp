
#include "hash_lock.hpp"

#include <immintrin.h> // _mm_pause()
#include <unistd.h>    // sysconf

// Size the map to the number of online processors: one spin slot per core.
HashLock::HashLock() : map(sysconf(_SC_NPROCESSORS_ONLN)) {}

void HashLock::lock() noexcept {
    // Grab a ticket and find the slot it hashes to.
    uint64_t my_ticket = next_ticket.fetch_add(1, std::memory_order_relaxed);
    HashNode* node = map.get_node(my_ticket);

    // Spin until the previous holder opens our slot. The acquire load pairs with
    // the release store in unlock() so the critical section can't float up above
    // the point where we actually own the lock.
    while (!node->your_turn.load(std::memory_order_acquire)) {
        _mm_pause();
    }

    // Consume the signal so the slot is reset for the next ticket that hashes
    // here (my_ticket + num_slots), and remember our ticket for unlock().
    node->your_turn.store(false, std::memory_order_relaxed);
    held_ticket = my_ticket;
}

void HashLock::unlock() noexcept {
    // Open the slot that the next ticket hashes to, handing it the lock.
    HashNode* node = map.get_node(held_ticket + 1);
    node->your_turn.store(true, std::memory_order_release);
}
