
#include "mcs_lock.hpp"
#include <immintrin.h> // _mm_pause()

thread_local MCSLock::Node local_node;

void MCSLock::lock() noexcept {
    Node* node = &local_node;
    node->next.store(nullptr, std::memory_order_relaxed);
    // Exchange tail: get predecessor
    Node* prev = tail.exchange(node, std::memory_order_acq_rel);
    if (prev != nullptr) {
        node->locked.store(true, std::memory_order_relaxed);
        // Link yourself
        prev->next.store(node, std::memory_order_release);
        // Spin on *local* flag
        while (node->locked.load(std::memory_order_acquire)) {
            _mm_pause();
        }
    }
    // else: no predecessor → lock acquired immediately
}

void MCSLock::unlock() noexcept {
    Node* node = &local_node;
    Node* succ = node->next.load(std::memory_order_acquire);
    if (succ == nullptr) {
        // Try to reset tail
        Node* expected = node;
        if (tail.compare_exchange_strong(
                expected, nullptr,
                std::memory_order_release,
                std::memory_order_relaxed)) {
            return;
        }
        // Wait until successor appears
        do {
            succ = node->next.load(std::memory_order_acquire);
            _mm_pause();
        } while (succ == nullptr);
    }
    // Pass lock to successor
    succ->locked.store(false, std::memory_order_release);
}
