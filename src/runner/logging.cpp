

#include "logging.hpp"
#include <vector>
#include <fstream>
#include <cstdio>
#include <cstdlib>

struct alignas(64) ThreadLog {
    LogEntry* data;
    size_t size;
    size_t dropped;
};

static std::vector<ThreadLog> all_logs;
size_t per_thread_capacity;

thread_local size_t idx;
thread_local size_t dropped;
thread_local LogEntry* buffer;

void logging_init(int num_threads, size_t capacity_per_thread) {
    per_thread_capacity = capacity_per_thread;
    all_logs.resize(num_threads);

    // initialize thread slots
    for (int i = 0; i < num_threads; i++) {
        all_logs[i].data = nullptr;
        all_logs[i].size = 0;
        all_logs[i].dropped = 0;
    }
}

void init_thread_log() {
    idx = 0;
    dropped = 0;
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
    all_logs[thread_id].dropped = dropped;
}

// Saturation is reported here rather than in the log file: dump_logs'
// (count, entries...) layout is what the Python LogParser reads, and changing it
// would break every existing parse. The first line is a stable machine-readable
// marker that scripts/runner.py greps for.
bool report_saturation() {
    size_t saturated_threads = 0;
    size_t total_dropped = 0;
    for (const ThreadLog& log : all_logs) {
        if (log.dropped > 0) {
            saturated_threads++;
            total_dropped += log.dropped;
        }
    }

    if (saturated_threads == 0) {
        return false;
    }

    std::fprintf(stderr,
                 "LOG_SATURATED threads=%zu/%zu capacity=%zu dropped=%zu\n",
                 saturated_threads, all_logs.size(), per_thread_capacity,
                 total_dropped);
    std::fprintf(stderr,
                 "WARNING: log buffer filled; %zu of %zu threads stopped recording "
                 "before the run ended.\n"
                 "  total_CS_completions is truncated to the capacity, and because "
                 "threads fill at\n"
                 "  different rates the fairness metrics understate unfairness. "
                 "Raise LOG_BUDGET_BYTES\n"
                 "  or shorten DURATION and re-run.\n",
                 saturated_threads, all_logs.size());
    return true;
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