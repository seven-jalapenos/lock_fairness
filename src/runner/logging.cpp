

#include "logging.hpp"
#include <vector>
#include <fstream>

struct alignas(64) ThreadLog {
    LogEntry* data;
    size_t size;
};

static std::vector<ThreadLog> all_logs;
size_t per_thread_capacity;

thread_local size_t idx;
thread_local LogEntry* buffer;

void logging_init(int num_threads, size_t capacity_per_thread) {
    per_thread_capacity = capacity_per_thread;
    all_logs.resize(num_threads);

    // initialize thread slots
    for (int i = 0; i < num_threads; i++) {
        all_logs[i].data = nullptr;
        all_logs[i].size = 0;
    }
}

void init_thread_log() {
    idx = 0;
    buffer = (LogEntry*) aligned_alloc(64, sizeof(LogEntry) * per_thread_capacity);
}

void finalize_thread_log(int thread_id) {
    all_logs[thread_id].data = buffer;
    all_logs[thread_id].size = idx;
}

void dump_logs(const std::string& file) {
    std::ofstream out(file, std::ios::binary);

    for (size_t t = 0; t < all_logs.size(); t++) {
        out.write((char*)&all_logs[t].size, sizeof(size_t));
        out.write((char*)all_logs[t].data,
                  all_logs[t].size * sizeof(LogEntry));
    }
}