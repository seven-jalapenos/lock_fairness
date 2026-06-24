
#pragma once
#include <atomic>
#include <cstdint>

#include "lockADT.hpp"

class TSSpinLock : public Lock {
private:
    std::atomic<bool> flag {false};
    std::atomic<uint64_t> next {UINT64_MAX};

public:
    void lock() noexcept;
    void unlock() noexcept;
};