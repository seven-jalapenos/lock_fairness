
# from itertools import product
# from runner import Runner

# param_space = {
#     'lock': ['mcs', 'clh', 'ticket', 'ttas', 'ttas_b'],
#     'threads': [1, 4, 8, 14, 28, 56],
#     'pin': [1]
# }

# log_dir = 'files/logs'

# def main():
#     keys = param_space.keys()
#     values = param_space.values()

#     for params in product(*values):
#         params_dict = dict(zip(keys, params))
#         output_dir = f"{log_dir}/{params_dict['lock']}_{params_dict['threads']}_{params_dict['pin']}"
#         print(f"starting runs with with params: {params_dict}")
#         # avg across 10 runs
#         for i in range(10):
#             runner = Runner(params_dict, output_dir, str(i))
#             runner()

# if __name__ == "__main__":
#     main()

from pathlib import Path

import argparse
import time
import faulthandler
import signal

from .all_metrics import average_all_metrics
from .plot_all import plot_single_runs, plot_cross_runs
from .runner import run_permutations, LOCK_TYPES

try:
    from .mail import send_email
except ImportError:
    # mail.py holds credentials and is gitignored, so it only exists on the
    # benchmark box. Degrade to a no-op rather than failing at import time --
    # otherwise even `--help` is unusable anywhere else.
    def send_email(subject: str, body: str, to_email: str) -> None:
        print(f"[mail disabled] {subject}", flush=True)


def parse_int_spec(values: list[str]) -> list[int]:
    """Parse a list of ints where any element may be an inclusive `A-B` range,
    so `--threads 1-28` and `--threads 1 4 8` are both accepted."""
    out = set()
    for value in values:
        if '-' in value.lstrip('-'):
            low, _, high = value.partition('-')
            try:
                start, end = int(low), int(high)
            except ValueError:
                raise argparse.ArgumentTypeError(f"invalid range: {value!r}")
            if start > end:
                raise argparse.ArgumentTypeError(f"empty range: {value!r}")
            out.update(range(start, end + 1))
        else:
            try:
                out.add(int(value))
            except ValueError:
                raise argparse.ArgumentTypeError(f"invalid integer: {value!r}")
    return sorted(out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python -m scripts.run_all',
        description='Run the lock fairness benchmark sweep and its analysis pipeline.'
    )
    parser.add_argument(
        '--locks', nargs='+', metavar='LOCK', choices=LOCK_TYPES,
        default=['mcs', 'clh', 'ticket', 'ttas', 'ttasb', 'tsspin'],
        help='lock implementations to sweep (choices: %(choices)s)'
    )
    parser.add_argument(
        '--threads', nargs='+', metavar='SPEC', default=['1-28'],
        help='thread counts, as values and/or inclusive ranges (e.g. "1-28" or "1 4 8")'
    )
    parser.add_argument(
        '--work', nargs='+', metavar='SPEC', default=['10000'],
        help='critical-section work sizes (loop iterations) to sweep'
    )
    parser.add_argument(
        '--pin', nargs='+', metavar='SPEC', default=['1'],
        help='core pinning policies: 0 none, 1 round-robin, 2 one hot core, 3 half hot core'
    )
    parser.add_argument(
        '--reps', type=int, default=10,
        help='runs per parameter combination, averaged together (default: %(default)s)'
    )
    parser.add_argument(
        '--out-dir', type=Path, default=Path('files/pqt_release'),
        help='parquet output tree (default: %(default)s)'
    )
    parser.add_argument(
        '--figures-dir', type=Path, default=Path('files/final_figures_2_release'),
        help='cross-run figure output directory (default: %(default)s)'
    )
    parser.add_argument(
        '--log-dir', type=Path, default=Path('files/logs'),
        help='raw binary log tree (default: %(default)s)'
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    space = {
        'lock': args.locks,
        'threads': parse_int_spec(args.threads),
        'pin': parse_int_spec(args.pin),
        'work': parse_int_spec(args.work),
    }

    # Dump a full Python traceback (all threads) on demand or on a hard crash.
    # If the sweep ever hangs again, from another shell on the box run:
    #     kill -USR1 $(pgrep -f scripts.run_all)
    # and the current stack — the exact line it's stuck on — prints to stderr,
    # without killing the process.
    faulthandler.enable()
    if hasattr(faulthandler, "register"):
        faulthandler.register(signal.SIGUSR1, all_threads=True)

    files_dir = args.out_dir
    figures_dir = args.figures_dir

    steps_completed = []

    start = time.time()

    print(f"sweeping {space} x {args.reps} reps", flush=True)

    try:
        dir_ids = run_permutations(
            csv_dir=str(files_dir),
            log_dir=str(args.log_dir),
            space=space,
            reps=args.reps
        )
        steps_completed.append("lock runs completed")

        # Steps 1 and 2 are scoped to this sweep's directories; they re-read every
        # parquet they touch, so there's no reason to redo unrelated earlier runs.
        # Step 3 is deliberately unscoped — it's the comparison figure.

        # Step 1: Average all metrics across runs and export to CSV
        average_all_metrics(files_dir, only=set(dir_ids))
        steps_completed.append("average metrics generated")

        # Step 2: Plot metrics for each individual run
        plot_single_runs(files_dir, only=set(dir_ids))
        steps_completed.append("single run metrics plotted")

        # Step 3: Plot comparative metrics across all runs
        plot_cross_runs(files_dir, figures_dir)
        steps_completed.append("cross run metrics plotted")

        end = time.time()
        duration = end - start
        duration_minutes = duration / 60
        duration_hours = duration_minutes / 60
        print(f"analysis completed in {duration_hours:.2f} hours", flush=True)
        send_email(
            subject="Lock Fairness Analysis Completed",
            body=f"The lock fairness analysis has completed successfully in {duration_hours:.2f} hours.",
            to_email="jcjurgen@go.olemiss.edu"
        )
    except Exception as e:
        # Surface the failure and how far we got instead of swallowing it (the
        # send_email notifier is disabled). Re-raise so the process exits non-zero.
        print(f"analysis FAILED: {e!r}", flush=True)
        print("steps completed before failure: " + ", ".join(steps_completed), flush=True)
        send_email(
            subject="Lock Fairness Analysis Failed",
            body=f"Analysis failed with error: {str(e)}\nSteps completed before failure: {'\n'.join(steps_completed)}",
            to_email="jcjurgen@go.olemiss.edu"
        )
        raise


if __name__ == '__main__':
    main()
