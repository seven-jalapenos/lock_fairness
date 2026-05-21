
#pragma once
#include <atomic>

#include "lockADT.hpp"

class MCSLock : public Lock {
public:
    struct Node {
        std::atomic<Node*> next{nullptr};
        std::atomic<bool> locked{false};
    };

private:
    alignas(64) std::atomic<Node*> tail{nullptr};

public:
    void lock() noexcept;
    void unlock() noexcept;

};