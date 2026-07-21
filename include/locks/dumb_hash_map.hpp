
#pragma once
#include <atomic>
#include <cstdint>
#include <memory>

// A single spin location that a waiting thread parks on. Aligned to a cache
// line so that threads spinning on neighbouring slots don't ping-pong the same
// line back and forth.
struct alignas(64) HashNode {
    std::atomic<bool> your_turn{false};
};

// A fixed-size "map" from ticket -> spin node. The requested slot count is
// rounded up to a power of two so that a ticket hashes to a slot with a single
// AND (ticket & mask) rather than a modulo/integer-division on the lock hot
// path. A ticket t always lands on slot (t & mask); with at least as many slots
// as contending threads, no two waiting threads ever share a slot, so this
// behaves like the flag array of Anderson's array lock. If more threads contend
// than there are slots (ticket t and t + num_slots collide) the mapping is no
// longer 1:1 and the lock breaks -- hence "dumb".
class DumbHashMap {
private:
    uint64_t mask; // num_slots - 1; num_slots is always a power of two
    std::unique_ptr<HashNode[]> nodes;

public:
    explicit DumbHashMap(int requested_slots);

    // Return the node that ticket hashes to.
    HashNode* get_node(uint64_t ticket) noexcept { return &nodes[ticket & mask]; }

    int size() const noexcept { return static_cast<int>(mask) + 1; }
};
