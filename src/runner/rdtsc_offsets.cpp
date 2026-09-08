
#include <vector>
#include <thread>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cinttypes> // PRId64
#include <sched.h>
#include <unistd.h>

#include "find_offsets.hpp"
#include "rdtscp.hpp" // For rdtscp implementation
#include "pin_thread.hpp" // For pin_thread_to_core implementation

std::string filename = "files/rdtsc_offsets.txt";

// One published TSC reading per core, padded to its own cache line. Without the
// padding eight cores share a line, so every publisher invalidates its
// neighbours' copies and the reference core's loads all miss -- which both slows
// the measurement and widens the very staleness the estimator is trying to
// minimize.
namespace {
struct alignas(64) TscSlot {
    std::atomic<uint64_t> value{0};
};

// True when the file already describes this machine, i.e. carries one entry per
// online core. A file from a box with a different core count is not reusable,
// and LogParser refuses it later anyway.
bool offsets_file_covers(const std::string& path, int num_cores) {
    FILE* f = fopen(path.c_str(), "r");
    if (f == nullptr) {
        return false;
    }
    char line[256];
    int found = 0;
    while (fgets(line, sizeof(line), f) != nullptr) {
        int core = 0;
        long long value = 0;
        if (sscanf(line, "Core %d: %lld cycles", &core, &value) == 2) {
            ++found;
        }
    }
    fclose(f);
    return found == num_cores;
}
} // namespace

// Measures each core's TSC offset relative to core 0 and writes them to
// `filename`, which the Python LogParser reads to make cross-core timestamps
// comparable.
//
// Skipped when a usable file is already on disk unless `force`, because the
// sweep driver re-measures on a fixed interval rather than paying for this on
// every one of thousands of runs.
void find_offsets(bool force) {
    const int num_cores = sysconf(_SC_NPROCESSORS_ONLN);
    // Each sample now brackets one remote load between two local TSC reads, so
    // it costs far more than the old bare load -- but it also yields a usable
    // bound, where the old scheme needed tens of millions of samples and still
    // returned noise. A few hundred thousand accepted samples per core is well
    // past the point where the bound stops tightening.
    const int iterations = 200000;
    // A bracket wider than this means the reference core was interrupted between
    // its two clock reads, so the sample says nothing about the offset.
    const uint64_t ACCEPT_WINDOW = 2000;

    if (!force && offsets_file_covers(filename, num_cores)) {
        return;
    }

    std::vector<TscSlot> tsc_values(num_cores);
    // Signed: a core whose TSC trails core 0 has a genuinely negative offset.
    // Computing this unsigned wrapped such a core to ~1.8e19, which is not a
    // minimum of anything and makes the Python parser overflow int64 outright.
    std::vector<int64_t> offsets(num_cores, 0);
    std::vector<char> sampled(num_cores, 0);

    std::atomic<int> ready_count{0};
    std::atomic<bool> start_flag{false};
    std::atomic<bool> done_flag{false};

    std::vector<std::thread> threads;

    for (int core = 0; core < num_cores; ++core) {
        threads.emplace_back([&, core]() {
            pin_thread_to_core(core);

            uint32_t aux;

            // Signal ready
            ready_count.fetch_add(1, std::memory_order_relaxed);

            // Wait for global start
            while (!start_flag.load(std::memory_order_acquire)) {}

            if (core != 0) {
                // Publish until the reference core says it is finished. A fixed
                // iteration count instead let the publishers exit early -- the
                // reference core does num_cores-1 remote loads per iteration and
                // so runs far longer -- after which it went on comparing against
                // frozen slots whose apparent staleness grew without bound. Those
                // samples dominated the minimum and made the offsets meaningless.
                while (!done_flag.load(std::memory_order_relaxed)) {
                    tsc_values[core].value.store(rdtscp(aux), std::memory_order_relaxed);
                    asm volatile("" ::: "memory");
                }
                return;
            }

            // Bracket every remote read between two local clock reads.
            //
            // Core c wrote t_other at some real instant T <= t1, so
            //     t_other = TSC_0(T) + offset_c <= t1 + offset_c,
            // i.e. (t_other - t1) is always a *lower bound* on offset_c, and it
            // reaches the true value as T approaches t1. Taking the maximum over
            // samples therefore converges on offset_c from below, and -- this is
            // the point -- a stale slot only makes t_other smaller, so a
            // descheduled publisher can never inflate the answer.
            //
            // The remaining failure is this core stalling between its own two
            // reads, which the bracket width detects directly; those samples are
            // dropped. Comparing a single bare read against the slot could not
            // separate the two cases: taking a minimum let one descheduled
            // publisher drag the offset megacycles negative, and taking a maximum
            // let one stall of *this* core drag it megacycles positive. Both were
            // observed on an idle 8-core box, differing by 10^7 cycles run to run.
            //
            // Residual bias: the bound converges from below, so it lands short by
            // however long the publisher's tightest store-to-load gap is -- about
            // one publisher loop period. On an idle 8-core box that reads as a
            // uniform -53 +/- 3 cycles across runs, well inside the
            // AMBIGUITY_CYCLES noise floor the analysis already reports against.
            for (int i = 0; i < iterations; ++i) {
                for (int c = 1; c < num_cores; ++c) {
                    uint64_t t0 = rdtscp(aux);
                    asm volatile("" ::: "memory");
                    uint64_t t_other = tsc_values[c].value.load(std::memory_order_relaxed);
                    asm volatile("" ::: "memory");
                    uint64_t t1 = rdtscp(aux);

                    if (t_other == 0) continue;          // c has not published yet
                    if (t1 - t0 > ACCEPT_WINDOW) continue; // we were interrupted

                    // Wrap-safe two's-complement difference: done unsigned, where
                    // overflow is defined, then reinterpreted signed so a core
                    // behind core 0 yields a negative value rather than ~1.8e19.
                    int64_t bound = static_cast<int64_t>(t_other - t1);

                    if (!sampled[c] || bound > offsets[c]) {
                        offsets[c] = bound;
                        sampled[c] = 1;
                    }
                }
            }
            done_flag.store(true, std::memory_order_relaxed);
        });
    }

    // Wait until all threads are ready
    while (ready_count.load(std::memory_order_acquire) < num_cores) {}

    // Start measurement
    start_flag.store(true, std::memory_order_release);

    for (auto &t : threads) {
        t.join();
    }

    // Core 0 is reference
    offsets[0] = 0;
    sampled[0] = 1;

    // A core that never published is a hole in the calibration, not a zero.
    // Say so rather than letting a sentinel reach the analysis as data.
    for (int i = 0; i < num_cores; ++i) {
        if (!sampled[i]) {
            offsets[i] = 0;
            fprintf(stderr,
                    "WARNING: core %d never published a TSC sample; its offset is "
                    "recorded as 0 and cross-thread timestamps involving it carry "
                    "unquantified error.\n", i);
        }
    }

    FILE* offset_file = fopen(filename.c_str(), "w");
    if (offset_file == nullptr) {
        fprintf(stderr,
                "find_offsets: cannot open %s for writing (does files/ exist?)\n",
                filename.c_str());
        abort();
    }

    for (int i = 0; i < num_cores; ++i) {
        fprintf(offset_file, "Core %d: %" PRId64 " cycles\n", i, offsets[i]);
    }
    fclose(offset_file);
}
