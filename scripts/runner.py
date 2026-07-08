
import os
import subprocess
import time
from itertools import product
import gc

from analysis import LogParser, DataExporter
from analysis import create_global_timeline

# rerun without mcs
param_space = {
    'lock': ['clh', 'ticket', 'ttas', 'ttasb'],
    'threads': [2, 4, 8, 14, 20, 26, 32, 38, 44, 50, 56],
    'pin': [1]
}

# param_space = {
#     'lock': ['mcs', 'clh', 'ticket', 'ttas', 'ttasb'],
#     'threads': [2, 4, 8, 14, 20, 26, 32, 38, 44, 50, 56],
#     'pin': [1]
# }

# param_space = {
#     'lock': ['ticket', 'ttas', 'ttas_b'],
#     'threads': [1, 4, 8, 14, 28, 56],
#     'pin': [1]
# }

log_dir = 'files/logs'
csv_dir = 'files/csv'

offset_file_path = 'files/rdtsc_offsets.txt'

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

        log_parser.close()
        pqt_writer.close()

def run_permutations():
    keys = param_space.keys()
    values = param_space.values()

    product_space = list(product(*values))
    missing = [
        ('mcs', 44, 1),
        ('mcs', 50, 1),
        ('mcs', 56, 1)
    ]
    product_space = missing + product_space
    # product_space.insert(0, ('clh', 56, 1)) # rerun 

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
            runner = Runner(params_dict, output_dir, csv_output_dir, str(i))
            runner()
            # The parse builds several large DataFrames per run; drop them and
            # force a collection so RSS doesn't ratchet up across the sweep and
            # trip the OOM killer on a later, larger run.
            del runner
            gc.collect()


if __name__ == "__main__":
    run_permutations()