
#pragma once
#include <cstdint>
#include <string>

struct LogEntry {
    uint64_t lock_invocation;    // caller thread lock() invocation
    uint64_t lock_aquisition;    // lock acquisition
    uint64_t lock_release;       // caller thread unlock()
};

void logging_init(int num_threads, const size_t capacity_per_thread);
void init_thread_log();
void finalize_thread_log(int thread_id);
void dump_logs(const std::string& file);

extern thread_local size_t idx;
extern thread_local LogEntry* buffer;
extern size_t per_thread_capacity;

inline void log_event(uint64_t lock_invocation, uint64_t lock_aquisition, uint64_t lock_release) {
    if (idx < per_thread_capacity) {
        buffer[idx++] = {lock_invocation,lock_aquisition, lock_release};
    }
}
