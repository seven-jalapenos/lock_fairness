
#include "dumb_hash_map.hpp"

#include <bit> // std::bit_ceil

DumbHashMap::DumbHashMap(int requested_slots) : mask(0), nodes(nullptr) {
    // Round the requested slot count up to a power of two so that get_node()
    // can mask (ticket & mask) instead of doing a runtime integer division.
    uint64_t slots =
        std::bit_ceil(static_cast<uint64_t>(requested_slots < 1 ? 1 : requested_slots));
    mask = slots - 1;
    nodes = std::make_unique<HashNode[]>(slots);

    // Ticket 0 hashes to slot 0, so slot 0 starts "open" to let the very first
    // ticket acquire the lock immediately. Every other slot starts closed.
    nodes[0].your_turn.store(true, std::memory_order_relaxed);
}
