
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

from .all_metrics import average_all_metrics
from .plot_all import plot_single_runs, plot_cross_runs
from .runner import run_permutations
from .mail import send_email
import time
import faulthandler
import signal

if __name__ == '__main__':
    # Dump a full Python traceback (all threads) on demand or on a hard crash.
    # If the sweep ever hangs again, from another shell on the box run:
    #     kill -USR1 $(pgrep -f scripts.run_all)
    # and the current stack — the exact line it's stuck on — prints to stderr,
    # without killing the process.
    faulthandler.enable()
    if hasattr(faulthandler, "register"):
        faulthandler.register(signal.SIGUSR1, all_threads=True)

    files_dir = Path("files/csv")
    figures_dir = Path("files/final_figures")

    steps_completed = []

    start = time.time()

    try:
        run_permutations()
        steps_completed.append("lock runs completed")

        # Step 1: Average all metrics across runs and export to CSV
        average_all_metrics(files_dir)
        steps_completed.append("average metrics generated")

        # Step 2: Plot metrics for each individual run
        plot_single_runs(files_dir)
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