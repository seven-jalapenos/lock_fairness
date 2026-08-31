
import os
import re
import subprocess
import time
from itertools import product
import gc

try:
    import psutil
except ImportError:
    # Only feeds the report_memory() debugging helper, and isn't in
    # requirements.txt -- not worth making the whole pipeline unimportable.
    psutil = None

from analysis import LogParser, DataExporter
from analysis import create_global_timeline

# Valid lock names, mirroring the dispatch in src/runner/main.cpp. Kept here so
# the CLI can reject a typo before burning a sweep on it.
LOCK_TYPES = ['mcs', 'clh', 'ticket', 'ttas', 'ttasb', 'tsspin', 'hash']

param_space = {
     'lock': ['mcs', 'clh', 'ticket', 'ttas', 'ttasb', 'tsspin'],
     'threads': list(range(1, 29)),
     'pin': [1],
     'work': [10000]
}


def run_dir_id(params: dict) -> str:
    """Canonical name for one parameter combination's output directory.

    Work size is part of the identity: without it, re-sweeping the same
    lock/threads/pin at a different CS length would collide with the previous
    result and be silently skipped by the resume check in _run_complete."""
    return f"{params['lock']}_{params['threads']}_{params['pin']}_w{params['work']}"

CMAKE_CACHE_PATH = 'build/CMakeCache.txt'


def _assert_release_build(cmake_cache_path: str = CMAKE_CACHE_PATH) -> None:
    """Refuse to run the sweep against a non-Release build.

    A Debug build (no -O3/-march=native) is easy to leave in place after a
    debugging session (cmake caches CMAKE_BUILD_TYPE until it's explicitly
    reconfigured) and silently produces a lock_exe that's an order of
    magnitude slower per iteration -- collapsing real contention and making
    every fairness metric from the sweep meaningless, with no error to signal
    it happened."""
    if not os.path.exists(cmake_cache_path):
        raise RuntimeError(
            f"{cmake_cache_path} not found -- configure the build first: "
            "cmake -B build -S . -DCMAKE_BUILD_TYPE=Release"
        )

    with open(cmake_cache_path) as f:
        cache = f.read()

    match = re.search(r'^CMAKE_BUILD_TYPE:STRING=(.*)$', cache, re.MULTILINE)
    build_type = match.group(1).strip() if match else ''

    if build_type != 'Release':
        raise RuntimeError(
            f"build/ is configured as CMAKE_BUILD_TYPE={build_type or '(empty)'}, not Release. "
            "Benchmark numbers from a non-Release build are not comparable (no -O3/-march=native) "
            "and will understate throughput/contention. Reconfigure and rebuild with: "
            "cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j"
        )


# log_dir = 'files/logs'
# csv_dir = 'files/csv'

offset_file_path = 'files/rdtsc_offsets.txt'

def report_memory():
    if psutil is None:
        return
    rss = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
    print(f"RSS: {rss:.2f} GiB", flush=True)


class Runner:
    # TODO 
    # runner is for one run
    # it should take params for the run, execute and then parse/write csv
    # implement a parameter checker

    def __init__(self, params: dict, output_dir: str, csv_dir: str, iteration_name: str):
        self.params = params
        self.output_dir = output_dir
        self.csv_dir = csv_dir
        self.iteration_name = iteration_name
        # Set by __call__ if lock_exe reported a filled log buffer.
        self.saturated = False
    
    def __call__(self) -> None:

        threads = self.params['threads']
        pin = self.params['pin']
        lock = self.params['lock']
        work = self.params['work']
        filename = f'{run_dir_id(self.params)}_{self.iteration_name}.bin'
        out_file = f'{self.output_dir}/{filename}'

        print(f"iteration {self.iteration_name} with params: {self.params} ...")

        # lock_exe is silent on both streams in the normal path (find_offsets writes
        # only to its file), so anything on stderr is worth echoing verbatim. The
        # saturation marker can't be recovered from the log file itself -- a
        # truncated log looks exactly like a short one.
        proc = subprocess.run(
            [
                './build/bin/lock_exe',
                str(threads),
                str(pin),
                lock,
                str(work),
                out_file
            ], 
            check=True,
            stderr=subprocess.PIPE,
            text=True
        )
        if proc.stderr:
            print(proc.stderr, end='', flush=True)
        self.saturated = 'LOG_SATURATED' in proc.stderr

        log_mb = os.path.getsize(out_file) / 1e6
        print(f"done (log {log_mb:.1f} MB)", flush=True)

        # Time the two parse stages separately and flush, so a hang here is
        # attributable to a concrete stage (read vs melt/sort) instead of the
        # vague "parsing logs..." — and so progress isn't lost to block buffering
        # when stdout is a tmux logfile rather than a TTY.
        print("parsing logs (read + calibrate)...", flush=True)
        t0 = time.perf_counter()
        log_parser = LogParser(out_file, offset_file_path, int(pin))
        assert(log_parser.all_threads_data is not None)
        n_events = len(log_parser.all_threads_data)
        print(f"  read {n_events} events in {time.perf_counter() - t0:.1f}s", flush=True)

        print("building global timeline (melt + sort)...", flush=True)
        t1 = time.perf_counter()
        global_timeline = create_global_timeline(log_parser.all_threads_data) # type: ignore
        print(f"  timeline of {len(global_timeline)} rows in {time.perf_counter() - t1:.1f}s", flush=True)

        print(f"writing raw data to {self.csv_dir} ...", flush=True)
        pqt_writer = DataExporter(
            log_parser.all_threads_data, # type: ignore
            global_timeline,
            self.csv_dir,
            self.iteration_name
        )
        pqt_writer.write_raw()
        print("done writing")

        log_parser.close()
        pqt_writer.close()

def _run_complete(csv_output_dir: str, run_name: str) -> bool:
    """Return True if this iteration's parquet outputs already exist AND are
    readable, so a resumed sweep can skip it. Both the flat data and the timeline
    parquet must be present and have a valid footer — a file truncated by a
    mid-write crash fails the footer read and is treated as incomplete (redone),
    so we never resume on top of a half-written, silently-bad file."""
    import pyarrow.parquet as pq

    paths = [
        os.path.join(csv_output_dir, 'data', f'{run_name}_data.parquet'),
        os.path.join(csv_output_dir, 'timeline', f'{run_name}_timeline.parquet'),
    ]
    for p in paths:
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            return False
        try:
            pq.read_metadata(p)  # reads only the footer; raises if truncated/corrupt
        except Exception:
            return False
    return True


def run_permutations(csv_dir: str, log_dir: str='files/logs',
                     space: dict | None = None,
                     reps: int = 10) -> tuple[list[str], list[str]]:
    """Sweep every combination in `space`, `reps` runs each.

    Returns (dir_ids, saturated). `dir_ids` are the run directory names produced,
    so the caller can scope the (expensive) averaging and per-run plotting to just
    this sweep instead of reprocessing every directory left behind by earlier ones.
    `saturated` names the runs whose log buffer filled -- their metrics are
    truncated and shouldn't be trusted."""
    _assert_release_build()

    if space is None:
        space = param_space

    keys = space.keys()
    values = space.values()

    product_space = list(product(*values))
    dir_ids = []
    saturated = []

    for params in product_space:
        params_dict = dict(zip(keys, params))

        dir_id = run_dir_id(params_dict)
        dir_ids.append(dir_id)
        output_dir = f"{log_dir}/{dir_id}"
        csv_output_dir = f"{csv_dir}/{dir_id}"

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(csv_output_dir, exist_ok=True)
        print(f"starting runs with with params: {params_dict}")
        # avg across `reps` runs
        for i in range(reps):
            # Resume/checkpoint: skip iterations whose parquet outputs already
            # exist and are valid, so a sweep killed partway (e.g. by the machine
            # crashing) can be re-launched and pick up where it left off instead
            # of recomputing everything.
            if _run_complete(csv_output_dir, str(i)):
                print(f"  skipping iteration {i} (already complete)", flush=True)
                continue
            runner = Runner(params_dict, output_dir, csv_output_dir, str(i))
            runner()
            if runner.saturated:
                saturated.append(f"{dir_id} iter {i}")
            # The parse builds several large DataFrames per run; drop them and
            # force a collection so RSS doesn't ratchet up across the sweep and
            # trip the OOM killer on a later, larger run.
            del runner
            gc.collect()

    # A warning printed 1600 runs ago isn't actionable; the list is. Only covers
    # runs actually executed here -- iterations skipped by the resume check above
    # were never re-run, so their saturation state is unknown.
    if saturated:
        print(f"\nWARNING: {len(saturated)} run(s) filled the log buffer and are "
              f"truncated (of those executed this invocation):", flush=True)
        for name in saturated:
            print(f"  {name}", flush=True)
        print("Raise LOG_BUDGET_BYTES or shorten DURATION and re-run these.\n",
              flush=True)

    return dir_ids, saturated


# if __name__ == "__main__":
#     run_permutations()
