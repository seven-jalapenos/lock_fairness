
#pragma once
#include <atomic>
#include <cstdint>

#include "lockADT.hpp"
#include "dumb_hash_map.hpp"

class HashLock : public Lock {
private:
    alignas(64) std::atomic<uint64_t> next_ticket{0};
    alignas(64) uint64_t held_ticket{0};
    DumbHashMap map;

public:
    HashLock();
    ~HashLock() override = default;

    void lock() noexcept override;
    void unlock() noexcept override;
};
