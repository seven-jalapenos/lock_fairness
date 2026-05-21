
#pragma once
#include <atomic>

#include "lockADT.hpp"

class TTASLock : public Lock {
private:
    std::atomic<bool> flag {false};

public:
    void lock() noexcept;
    void unlock() noexcept;
};