# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A benchmarking harness for studying **fairness** of mutual-exclusion lock algorithms (MCS, CLH, ticket, TTAS, TTAS w/ backoff, timestamp-based spin locks) under contention. A C++ benchmark pins threads to cores, hammers a chosen lock implementation, and logs per-acquisition `rdtscp` timestamps to a binary file. A Python analysis pipeline parses those binary logs, calibrates timestamps across cores, computes fairness metrics (wait time, overtakes, rank-inversion penalty, lock-transfer matrix), and produces CSV/parquet/figures.

## Build (C++)

CMake + C++20, requires pthreads. Release flags use `-O3 -march=native` (must build on the actual benchmark machine, not cross-compiled).

```bash
cmake -B build -S .
cmake --build build -j
```

Produces `build/bin/lock_exe`. There are no C++ unit tests — validation is done by running the benchmark and inspecting logs/metrics.

Run the benchmark directly:

```bash
./build/bin/lock_exe [num_threads] [core_pin_policy] [lock_type] [work] [filename(optional)]
```

- `core_pin_policy`: `0` = no pinning, `1` = round-robin, `2` = all threads on one core, `3` = half on one hot core / half round-robin.
- `lock_type`: `mcs`, `clh`, `ticket`, `ttas`, `ttasb`, `tsspin`, `hash` (must match a branch in `src/runner/main.cpp`'s lock-type dispatch — new locks need a line added there).
- `work`: number of loop iterations of simulated work inside the critical section. Rejected unless it's a non-negative integer — `argv[4]` used to be the filename, so a stale 4-arg invocation errors out instead of silently running zero work.
- Run with all args or no args (defaults: 8 threads, round-robin pin, MCS, 10000 iterations).
- `DURATION`/`WARMUP`/`CAPACITY` (10s run, 3s warmup, ~8.4M log entries/thread) are compile-time constants in `main.cpp`. A thread that exceeds `CAPACITY` stops logging rather than growing memory, so short critical sections can saturate the buffer before `DURATION` elapses.

Every run also (re)computes per-core `rdtscp` calibration offsets via `find_offsets()` and writes them to `files/rdtsc_offsets.txt` before starting — the Python side depends on this file to make cross-core timestamps comparable.

## Python analysis pipeline

Python 3.14, dependencies in `requirements.txt` (numpy/pandas/pyarrow/matplotlib), installed in `venv/`. Activate with `source venv/bin/activate`.

Full pipeline (run benchmark sweep, average metrics, plot single-run and cross-run figures), invoked as a module from repo root:

```bash
python -m scripts.run_all
python -m scripts.run_all --locks mcs ticket --threads 1-8 16 --work 100 10000 --reps 3
```

Flags (all optional; defaults reproduce the previous hardcoded sweep): `--locks`, `--threads`, `--work`, `--pin`, `--reps`, `--out-dir`, `--figures-dir`, `--log-dir`. `--threads`/`--work`/`--pin` accept both bare values and inclusive `A-B` ranges.

This calls, in order: `scripts.runner.run_permutations()` (drives `build/bin/lock_exe` across the lock/thread-count/pin-policy/work grid built from those flags, defaulting to `scripts/runner.py`'s `param_space`, `--reps` iterations each), `analysis` metric averaging, then single-run and cross-run plotting.

Metric averaging and single-run plotting are **scoped to the directories the sweep just produced** (they re-read every parquet they touch, so a narrow sweep shouldn't redo the whole tree). Cross-run plotting is deliberately **unscoped**, so a sweep of one lock is still drawn against everything previously measured under `--out-dir`, and it emits one figure set per work size (`cross_run_<metric>_by_threads_w<work>.png`) since runs at different CS lengths don't belong on the same line. There's no test suite for the Python code either; sanity-check by running against a small param space and inspecting the output CSVs/figures.

`scripts/run_all.py` is mid-refactor: the top block (old `product`-based driver) is dead code left commented out — don't resurrect it, the live path is the `run_permutations`/`average_all_metrics`/`plot_*` sequence below it.

## Architecture

### C++ side (`include/locks/`, `src/locks/`, `include/runner/`, `src/runner/`)

- **`Lock` ADT** (`include/locks/lockADT.hpp`): abstract base with `lock()`/`unlock()` (both `noexcept`), non-copyable. Every lock implementation subclasses this. Note there is a duplicate, slightly older copy at `src/locks/lockADT.hpp` (no `noexcept`, no copy-delete) — the one under `include/locks/` is the one actually wired into the CMake include path and used by lock headers; treat the `src/` copy as stale/unused.
- Each lock is a header (class + inline/declared members) in `include/locks/*.hpp` with the implementation in `src/locks/*.cpp`. New locks must be added to `src/locks/CMakeLists.txt`'s `add_library(locks STATIC ...)` list and to the `lock_type` dispatch in `src/runner/main.cpp` to be usable from the CLI.
- Currently **wired into the build**: `clh`, `mcs`, `ticket`, `ttas`, `ttas_backoff`, `tsspin`.
- **Untracked/in-progress locks** (`selection_lock`, `tsh_lock`, `tst_lock` — currently new/untracked files, not yet in `CMakeLists.txt` or `main.cpp`): tree/timestamp-based selection lock designs. `tsh_lock.hpp` and `tst_lock.hpp` currently have a bare `traverse();` declaration with no type — this doesn't compile; expect to fix this when picking the work back up.
- `include/runner/logging.hpp`: lock-free-ish thread-local ring buffer logging. Each worker thread gets its own preallocated `LogEntry` buffer (`invocation`/`acquisition`/`release` timestamps); `dump_logs()` writes one `(count, entries...)` block per thread to the binary log file, in thread-spawn order. The Python `LogParser` depends on this exact binary layout (see below).
- `include/runner/pin_thread.hpp`: `pthread_setaffinity_np` wrapper, aborts on invalid core id.
- `src/runner/rdtsc_offsets.cpp` (`find_offsets()`): measures per-core TSC offsets relative to core 0 by racing threads pinned to every core and taking the minimum observed diff; writes `files/rdtsc_offsets.txt` as `Core N: X cycles` lines.
- `src/runner/main.cpp` worker loop: barrier-sync all threads → busy-spin warmup phase (repeatedly lock/unlock until `start` flag) → benchmark phase (record `rdtsc` before `lock()`, `rdtscp` after acquire and before `unlock()`, plus a fixed-iteration busy-work loop as the simulated critical section) until `stop` flag. Timestamps use `_mm_lfence`/compiler barriers around the TSC reads to bound reordering.

### Python side (`analysis/`, `scripts/`)

Pipeline stages, each a module under `analysis/`, wired together via `analysis/__init__.py`:

1. **`LogParser`** (`log_parser.py`) — reads the per-thread binary log (`size_t count` + `count` × `(invocation, acquisition, release)` u64 triples), applies per-core TSC offset calibration (`calibrate_log`, using `pinning_policy` to map thread id → core id — mirrors the pinning logic in `main.cpp`), and produces one flat per-event `DataFrame` (`thread_id`, timestamps, derived `wait_time`/`hold_time`). Pinning policy `3` calibration is a known TODO (unimplemented / marked in code).
2. **`create_global_timeline`**(`log_parser.py`) — melts the flat per-event frame into a single chronologically-sorted stream of `(thread_id, event_type, timestamp)` used for cross-thread ordering analysis (overtakes, lock transfers).
3. **`LogAnalyzer`** (`log_analyzer.py`) — computes fairness metrics from one run's data + global timeline:
   - `create_overtake_timeline()`: single-pass O(N·t) chronological scan tracking pending lock invocations; when a thread acquires the lock, every other thread that invoked earlier but is still pending counts as "overtaken" (an intervening acquisition).
   - Per-thread metrics: `find_avg_per_thread_wait_time`, `percent_time_in_CS`.
   - Global metrics: `overtake_percentage`, `average_overtake_depth`, `mean_squared_rank_inversion_penalty` (RMS of overtake counts — quadratic penalty so one thread overtaken 3× is weighted worse than 3 threads overtaken once), `total_CS_completions`, `lock_transfer_matrix` (n×n count of which thread acquires immediately after which).
4. **`DataExporter`** (`data_exporter.py`) — writes per-run flat data + global timeline to parquet under `<csv_dir>/<dir_id>/{data,timeline}/`.
5. **`MetricAverager`** (`metric_averager.py`) — pools per-run metrics across the `--reps` iterations of a given parameter combination into `Stats(avg, std)` (`analysis/defs.py`). Uses proper pooled-variance formulas (within-group + between-group sum of squares) rather than naively averaging standard deviations — this matters when combining runs with different sample counts. Rank-inversion penalty is pooled in squared space then converted to RMS via the delta method for the stddev.
6. **`StatsExporter`**, **`SingleRunPlotter`**, **`CrossRunPlotter`** — write averaged metrics to CSV and render figures (`files/final_figures/`), driven by `scripts/all_metrics.py` / `scripts/plot_all.py`.

`scripts/runner.py`'s `Runner` class drives one benchmark invocation end-to-end (spawn `lock_exe` subprocess → parse binary log → export parquet); `run_permutations()` sweeps a param-space grid × `reps` iterations, writing to `files/logs/<dir_id>/` (raw binary) and the parquet tree at `<csv_dir>/<dir_id>/`, and returns the `dir_id`s it produced so the caller can scope the analysis stages.

`dir_id` is `<lock>_<threads>_<pin>_w<work>`, built by `run_dir_id()` — the single definition of the naming, used for both the directory and the `.bin` basename. Work size is part of the identity on purpose: `_run_complete()` resumes a killed sweep by checking whether a combination's parquet already exists, so without the work suffix a re-sweep at a different CS length would collide with the old result and be silently skipped. `CrossRunPlotter._default_param_parser` parses this name back into params with an anchored regex; directories from before the work dimension (`<lock>_<threads>_<pin>`) still parse, with `work` as `None`.

### Data flow summary

```
lock_exe (C++) --binary log--> LogParser --DataFrame--> LogAnalyzer --metrics--> MetricAverager --Stats--> StatsExporter/Plotters
                             \-> files/rdtsc_offsets.txt (per-core calibration, read by LogParser)
```

`files/` (logs, csv, figures, rdtsc offsets) is entirely generated/gitignored — never assume it's checked in; regenerate via the pipeline above. `files/final_figures/` and `files/final_figures1/` in the working tree are prior output snapshots, not source.
