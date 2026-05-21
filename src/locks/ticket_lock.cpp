

#include "ticket_lock.hpp"
#include <immintrin.h> // for _mm_pause()


void TicketLock::lock() noexcept {
    uint32_t my_ticket = next_ticket.fetch_add(1, std::memory_order_relaxed);

    while (true) {
        uint32_t current = now_serving.load(std::memory_order_acquire);
        if (current == my_ticket)
            break;

        uint32_t distance = my_ticket - current;

        for (uint32_t i = 0; i < distance * 50; ++i)
            _mm_pause();
    }
}

void TicketLock::unlock() noexcept {
    uint32_t next = now_serving.load(std::memory_order_relaxed) + 1;
    now_serving.store(next, std::memory_order_release);
}
