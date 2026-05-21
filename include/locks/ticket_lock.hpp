
#pragma once
#include <atomic>

#include "lockADT.hpp"

class TicketLock : public Lock {
private:
    alignas(64) std::atomic<uint32_t> next_ticket{0};
    alignas(64) std::atomic<uint32_t> now_serving{0};

public:
    void lock() noexcept;
    void unlock() noexcept;
};