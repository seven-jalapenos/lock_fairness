

#include "logging.hpp"
#include <vector>
#include <fstream>
#include <cstdio>
#include <cstdlib>

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
    const size_t bytes = sizeof(LogEntry) * per_thread_capacity;
    buffer = (LogEntry*) aligned_alloc(64, bytes);
    // Fail loudly instead of segfaulting later in log_event: under memory
    // pressure aligned_alloc can return nullptr, and log_event only guards the
    // index bound, not a null buffer.
    if (buffer == nullptr) {
        std::fprintf(stderr,
                     "init_thread_log: failed to allocate %zu-byte log buffer\n",
                     bytes);
        std::abort();
    }
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
        // Release the per-thread buffer now that it is on disk. Harmless for the
        // process's own exit, but keeps peak RSS honest if dump_logs is ever
        // called before the process tears down.
        free(all_logs[t].data);
        all_logs[t].data = nullptr;
    }
}