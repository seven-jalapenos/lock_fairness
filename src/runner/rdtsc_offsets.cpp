
#include <vector>
#include <thread>
#include <atomic>
#include <cstdint>
#include <limits>
#include <sched.h>
#include <unistd.h>

#include "find_offsets.hpp"
#include "rdtscp.hpp" // For rdtscp implementation
#include "pin_thread.hpp" // For pin_thread_to_core implementation

std::string filename = "files/rdtsc_offsets.txt";

void find_offsets() {
    const int num_cores = sysconf(_SC_NPROCESSORS_ONLN);
    const int iterations = 10000000;

    std::vector<std::atomic<uint64_t>> tsc_values(num_cores);
    std::vector<uint64_t> offsets(num_cores, std::numeric_limits<uint64_t>::max());

    std::atomic<int> ready_count{0};
    std::atomic<bool> start_flag{false};

    std::vector<std::thread> threads;

    for (int core = 0; core < num_cores; ++core) {
        threads.emplace_back([&, core]() {
            pin_thread_to_core(core);

            uint32_t aux;

            // Signal ready
            ready_count.fetch_add(1, std::memory_order_relaxed);

            // Wait for global start
            while (!start_flag.load(std::memory_order_acquire)) {}

            for (int i = 0; i < iterations; ++i) {
                // Read TSC
                uint64_t t = rdtscp(aux);
                tsc_values[core].store(t, std::memory_order_relaxed);

                // Small fence to reduce reordering noise
                asm volatile("" ::: "memory");

                // Reference core computes offsets
                if (core == 0) {
                    uint64_t t_ref = t;

                    for (int c = 1; c < num_cores; ++c) {
                        uint64_t t_other = tsc_values[c].load(std::memory_order_relaxed);

                        if (t_other == 0) continue;

                        uint64_t diff = t_other - t_ref;

                        if (diff < offsets[c]) {
                            offsets[c] = diff;
                        }
                    }
                }
            }
        });
    }

    // Wait until all threads are ready
    while (ready_count.load(std::memory_order_acquire) < num_cores) {}

    // Start measurement
    start_flag.store(true, std::memory_order_release);

    for (auto &t : threads) {
        t.join();
    }

    FILE* offset_file = fopen(filename.c_str(), "w");
    // Core 0 is reference
    offsets[0] = 0;

    for (int i = 0; i < num_cores; ++i) {
        fprintf(offset_file, "Core %d: %lu cycles\n", i, offsets[i]);
    }
    if (offset_file) {
        fclose(offset_file);
    }
}
