
import os
import subprocess
import time
from itertools import product
import psutil
import gc

from analysis import LogParser, DataExporter
from analysis import create_global_timeline

param_space = {
     'lock': ['mcs', 'clh', 'ticket', 'ttas', 'ttasb', 'tsspin'],
     'threads': list(range(1, 29)),
     'pin': [1]
}


# log_dir = 'files/logs'
# csv_dir = 'files/csv'

offset_file_path = 'files/rdtsc_offsets.txt'

process = psutil.Process(os.getpid())

def report_memory():
    rss = process.memory_info().rss / (1024 ** 3)
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
    
    def __call__(self) -> None:

        threads = self.params['threads']
        pin = self.params['pin']
        lock = self.params['lock']
        filename = f'{lock}_{threads}_{pin}_{self.iteration_name}.bin'
        out_file = f'{self.output_dir}/{filename}'

        print(f"iteration {self.iteration_name} with params: {self.params} ...")

        subprocess.run(
            [
                './build/bin/lock_exe',
                str(threads),
                str(pin),
                lock,
                out_file
            ], 
            check=True
        )

        log_mb = os.path.getsize(out_file) / 1e6
        print(f"done (log {log_mb:.1f} MB)", flush=True)

        # Time the two parse stages separately and flush, so a hang here is
        # attributable to a concrete stage (read vs melt/sort) instead of the
        # vague "parsing logs..." — and so progress isn't lost to block buffering
        # when stdout is a tmux logfile rather than a TTY.
        print("parsing logs (read + calibrate)...", flush=True)
        t0 = time.perf_counter()
        log_parser = LogParser(out_file, offset_file_path, int(pin))
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


def run_permutations(csv_dir: str, log_dir: str='files/logs'):
    keys = param_space.keys()
    values = param_space.values()

    product_space = list(product(*values))

    for params in product_space:
        params_dict = dict(zip(keys, params))

        dir_id = f"{params_dict['lock']}_{params_dict['threads']}_{params_dict['pin']}"
        output_dir = f"{log_dir}/{dir_id}"
        csv_output_dir = f"{csv_dir}/{dir_id}"

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(csv_output_dir, exist_ok=True)
        print(f"starting runs with with params: {params_dict}")
        # avg across 10 runs
        for i in range(10):
            # Resume/checkpoint: skip iterations whose parquet outputs already
            # exist and are valid, so a sweep killed partway (e.g. by the machine
            # crashing) can be re-launched and pick up where it left off instead
            # of recomputing everything.
            if _run_complete(csv_output_dir, str(i)):
                print(f"  skipping iteration {i} (already complete)", flush=True)
                continue
            runner = Runner(params_dict, output_dir, csv_output_dir, str(i))
            runner()
            # The parse builds several large DataFrames per run; drop them and
            # force a collection so RSS doesn't ratchet up across the sweep and
            # trip the OOM killer on a later, larger run.
            del runner
            gc.collect()


# if __name__ == "__main__":
#     run_permutations()
