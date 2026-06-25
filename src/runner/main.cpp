
#include <emmintrin.h>
#include <iterator>
#include <memory>
#include <string>
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

#define COMPILER_BARRIER() asm volatile("" ::: "memory")

constexpr int CAPACITY = 1 << 26; // capacity of per-thread log buffer, ~67 million
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
// look at simulated work, I will probably need to change it to something else if it is optimized away by the compiler
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

        // simulated work
        int x = 0;
        for (int i = 0; i < iterations; i++) {
            x += i;
        }
        asm volatile("" :: "r"(x));

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
//      ARGS: [num_threads(>0)] [core_pin_policy] [lock_type] [iteration number] [filename(optional)]
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
//                 ttas_b (ttas with backoff)
// 
//      iteration number: number of iterations in critical section
//
//      filename: name of output binary log file (optional, if not provided, will be generated based on other parameters)
//
//      Run with all or no arguments
//      If run with no arguments, defaults to 8 threads, round-robin pinning, MCS lock and 1000 iterations.
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
    // set filename
    if (argc == 5) {
        filename = argv[4];
    } else {
        filename = LOG_DIR + "log_" + lock_type +
                   "_" + std::to_string(num_threads) +
                   "threads_pin" + std::to_string(pin) + ".bin";
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
    } else if (lock_type == "ttas_b") {
        lock = std::make_unique<TTASLock_Backoff>();
    } else if (lock_type == "tsspin"){
        lock = std::make_unique<TSSpinLock>();
    }
    else {
        std::cerr << "Unknown lock type: " << lock_type << "\n";
        return 1;
    }

    // find rtsc offsets for each core and save to file (used for post-processing logs)
    find_offsets(); // turn off for debugging

    sync_point = std::make_unique<std::barrier<>>(num_threads);
    logging_init(num_threads, CAPACITY);

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

    dump_logs(filename);

    // std::cout << "Done\n";
    return 0;
}