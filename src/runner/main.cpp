
#include <algorithm>
#include <emmintrin.h>
#include <iterator>
#include <memory>
#include <string>
#include <stdexcept>
#include <sys/types.h>
#include <thread>
#include <vector>
#include <iostream>
#include <barrier>

#include <x86intrin.h> // _mm_lfence, __rdtsc, and __rdtscp 

#include "logging.hpp"
#include "find_offsets.hpp" // for finding rdtscp offsets
#include "pin_thread.hpp"

#include "mcs_lock.hpp"
#include "clh_lock.hpp"
#include "ticket_lock.hpp"
#include "ttas_lock.hpp"
#include "ttas_backoff_lock.hpp"
#include "tsspin_lock.hpp"
#include "hash_lock.hpp"

#define COMPILER_BARRIER() asm volatile("" ::: "memory")

// Log buffer sizing is a TOTAL budget, not a per-thread constant. Total events in
// a run is set by CS length and DURATION -- the lock serializes, so thread count
// barely moves it -- while a per-thread constant allocates in proportion to
// threads. That is inverted: 1-thread runs saturate at 8.4M entries while
// reserving 200 MiB, and 56-thread runs reserve 11 GiB they never fill.
// Overridable at build time (-DLOG_BUDGET_BYTES=...) so a box can tune it without
// editing source.
#ifndef LOG_BUDGET_BYTES
#define LOG_BUDGET_BYTES (12ULL << 30)   // 12 GiB, about what 56 threads reserved before
#endif
// Ceiling so a 1- or 2-thread run doesn't reserve the whole budget. Measured on
// this box, a single thread turns over ~1.86e11/(work+1200) acquisitions in 10s,
// so the worst case (work=0) is ~155M events; 1<<28 leaves ~70% headroom. Actual
// RSS stays far below the reservation -- total events is roughly independent of
// thread count, so touched memory peaks near 155M*24B (~3.7 GiB) no matter how
// the budget is divided.
constexpr size_t MAX_CAPACITY = 1 << 28; // ~268M entries, ~6.4 GiB
constexpr size_t MIN_CAPACITY = 1 << 20; // floor for very high thread counts
constexpr int DURATION = 10; // seconds
constexpr int WARMUP = 3; // seconds
std::string LOG_DIR = "../files/logs/";

std::atomic<bool> start = false;
std::atomic<bool> stop = false;
std::unique_ptr<std::barrier<>> sync_point; // will be re-initialized in main with correct number of threads


// Worker thread function
// each thread will perform the following steps:
// 1. sync on barrier
// 2. warmup phase with nops until main thread signals start
// 3. benchmark phase: repeatedly acquire the lock, simulate work, release lock
//   - timestamps are recorded immediately before lock invocation, immediately after lock acquisition, and immediately before lock release
// Simulated critical section. The empty asm consumes x so the loop survives -O3;
// `iterations` is the CS length knob driven from the command line.
static inline __attribute__((always_inline)) void simulate_work(int iterations) {
    int x = 0;
    for (int i = 0; i < iterations; i++) {
        x += i;
    }
    asm volatile("" :: "r"(x));
}

void worker(int thread_id, int core_id, Lock* lock, int iterations) {
    init_thread_log();
    if (core_id >= 0) {
        pin_thread_to_core(core_id);
    }
    uint32_t aux;
    // phase 0: sync threads
    sync_point->arrive_and_wait();
    // phase 1: warmup
    while(!start.load(std::memory_order_relaxed)) {
        lock->lock();
        COMPILER_BARRIER();
        simulate_work(iterations);
        COMPILER_BARRIER();
        lock->unlock();
    }
    // phase 2: benchmark
    while(!stop.load(std::memory_order_relaxed)) {
        _mm_lfence();
        COMPILER_BARRIER();
        uint64_t lock_invoke = __rdtsc();   // timestamp before lock invocation
        COMPILER_BARRIER();

        lock->lock();

        COMPILER_BARRIER();
        uint64_t lock_acquire = __rdtscp(&aux);  // timestamp immediately after lock acquisition
        COMPILER_BARRIER();
        _mm_lfence();

        simulate_work(iterations);

        COMPILER_BARRIER();
        uint64_t lock_release = __rdtscp(&aux);  // timestamp immediately before lock release
        COMPILER_BARRIER();
        _mm_lfence();
        
        lock->unlock();        

        log_event(lock_invoke, lock_acquire, lock_release);
    }

    finalize_thread_log(thread_id);
}

/////////////////////////////////////////////////////////////
//
//      ARGS: [num_threads(>0)] [core_pin_policy] [lock_type] [work] [filename(optional)]
//      
//      num_threads: number of worker threads to spawn
//      core_pin_policy: 0 (no pinning)
//                       1 (round-robin pinning)
//                       2 (pin all threads to one core)
//                       3 (1/2 threads on core 0, 1/2 round-robin on remaining cores)
//      lock_type: mcs
//                 clh
//                 ticket
//                 ttas
//                 ttasb (ttas with backoff)
//                 tsspin
//                 hash
// 
//      work: number of iterations of busy-work in the critical section
//
//      filename: name of output binary log file (optional, if not provided, will be generated based on other parameters)
//
//      Run with all or no arguments
//      If run with no arguments, defaults to 8 threads, round-robin pinning, MCS lock and 10000 iterations.
//
//

int main(int argc, char* argv[]) {
    int num_threads = 8;
    int pin = 1;
    std::string lock_type = "mcs";
    int work = 10000;
    std::string filename;
    // parse arguments
    if (argc >= 4) {
        num_threads = std::atoi(argv[1]);
        pin = std::atoi(argv[2]);
        lock_type = argv[3];
    } 
    // argv[4] used to be the output filename; it is now the CS work size. Reject a
    // non-numeric value outright rather than letting atoi turn a stale 4-arg
    // invocation into a silent zero-work run written to the fallback path.
    if (argc >= 5) {
        std::string work_arg = argv[4];
        size_t consumed = 0;
        bool valid = !work_arg.empty();
        if (valid) {
            try {
                work = std::stoi(work_arg, &consumed);
            } catch (const std::exception&) {
                valid = false;
            }
        }
        if (!valid || consumed != work_arg.size() || work < 0) {
            std::cerr << "Invalid work argument: " << work_arg << "\n"
                      << "Usage: lock_exe [num_threads] [core_pin_policy] [lock_type] [work] [filename(optional)]\n";
            return 1;
        }
    }
    // set filename
    if (argc >= 6) {
        filename = argv[5];
    } else {
        filename = LOG_DIR + "log_" + lock_type +
                   "_" + std::to_string(num_threads) +
                   "threads_pin" + std::to_string(pin) +
                   "_w" + std::to_string(work) + ".bin";
    }

    // set pinning policy and generate core ids for each thread
    // -1 means no pinning
    std::vector<std::thread> threads;
    std::vector<int> core_ids(num_threads, -1);
    if (pin == 1) {
        for (int i = 0; i < num_threads; i++) {
            core_ids[i] = i % sysconf(_SC_NPROCESSORS_ONLN);
        }
    } else if (pin == 2) {
        int hot_core = sysconf(_SC_NPROCESSORS_ONLN) / 2;
        for (int i = 0; i < num_threads; i++) {
            core_ids[i] = hot_core;
        }
    } else if (pin == 3) {
        int hot_core = sysconf(_SC_NPROCESSORS_ONLN) / 2;
        int half = num_threads / 2;

        for (int i = 0; i < half; i++) {
            core_ids[i] = hot_core;
        }

        for (int i = half; i < num_threads; i++) {
            core_ids[i] = (i - half) % sysconf(_SC_NPROCESSORS_ONLN);
        }
    }

    // set lock type
    std::unique_ptr<Lock> lock;
    if (lock_type == "mcs") {
        lock = std::make_unique<MCSLock>();
    } else if (lock_type == "clh") {
        lock = std::make_unique<CLHLock>();
    } else if (lock_type == "ticket") {
        lock = std::make_unique<TicketLock>();
    } else if (lock_type == "ttas") {
        lock = std::make_unique<TTASLock>();
    } else if (lock_type == "ttasb") {
        lock = std::make_unique<TTASLock_Backoff>();
    } else if (lock_type == "tsspin"){
        lock = std::make_unique<TSSpinLock>();
    } else if (lock_type == "hash") {
        lock = std::make_unique<HashLock>();
    }
    else {
        std::cerr << "Unknown lock type: " << lock_type << "\n";
        return 1;
    }

    // find rtsc offsets for each core and save to file (used for post-processing logs)
    find_offsets(); // turn off for debugging

    sync_point = std::make_unique<std::barrier<>>(num_threads);
    size_t per_thread = LOG_BUDGET_BYTES / (sizeof(LogEntry) * (size_t)num_threads);
    per_thread = std::min(MAX_CAPACITY, std::max(MIN_CAPACITY, per_thread));
    logging_init(num_threads, per_thread);

    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back(worker, i, core_ids[i], lock.get(), work);
    }
    // warmup phase
    std::this_thread::sleep_for(std::chrono::seconds(WARMUP));
    start.store(true, std::memory_order_relaxed);
    // benchmark phase
    std::this_thread::sleep_for(std::chrono::seconds(DURATION)); // microseconds for debugging
    stop.store(true, std::memory_order_relaxed);

    for (auto& t : threads) {
        t.join();
    }

    // Warn before writing: a saturated run's log is truncated, which the Python
    // side cannot detect from the file alone. Exit status stays 0 on purpose --
    // the sweep driver uses check=True and shouldn't die over one bad combination.
    report_saturation();

    dump_logs(filename);

    // std::cout << "Done\n";
    return 0;
}