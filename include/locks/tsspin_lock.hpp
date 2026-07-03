
#pragma once
#include <atomic>
#include <cstdint>

#include "lockADT.hpp"

class TSSpinLock : public Lock {
private:
    alignas(64) std::atomic<bool> flag {false};
    alignas(64) std::atomic<uint64_t> next {UINT64_MAX};

public:
    void lock() noexcept;
    void unlock() noexcept;
};