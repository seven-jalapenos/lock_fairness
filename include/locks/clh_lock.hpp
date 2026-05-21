
#pragma once
#include <atomic>

#include "lockADT.hpp"

class CLHLock : public Lock {
public:
    struct Node {
        std::atomic<bool> locked;
    };

private:
    alignas(64) std::atomic<Node*> tail;
public:
    CLHLock();
    void lock() noexcept;
    void unlock() noexcept;
};
