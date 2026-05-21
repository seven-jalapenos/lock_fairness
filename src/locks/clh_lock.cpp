
#include "clh_lock.hpp"
#include <immintrin.h> // _mm_pause()

typedef CLHLock::Node Node;
thread_local Node* my_node = nullptr;
thread_local Node* my_pred = nullptr;

CLHLock::CLHLock() {
    // Dummy node so first thread has a predecessor
    Node* dummy = new Node{false};
    tail.store(dummy, std::memory_order_relaxed);
}

void CLHLock::lock() noexcept {
    if (my_node == nullptr) {
        my_node = new Node{false};
    }
    my_node->locked.store(true, std::memory_order_relaxed);
    // Swap into tail, get predecessor
    my_pred = tail.exchange(my_node, std::memory_order_acq_rel);
    // Spin on predecessor’s flag
    while (my_pred->locked.load(std::memory_order_acquire)) {
        _mm_pause();
    }
}

void CLHLock::unlock() noexcept {
    my_node->locked.store(false, std::memory_order_release);
    // Reuse predecessor node to avoid allocation
    my_node = my_pred;
}
