
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

from all_metrics import average_all_metrics
from plot_all import plot_single_runs, plot_cross_runs

if __name__ == '__main__':
    files_dir = Path("files/csv")
    figures_dir = Path("files/final_figures")

    # Step 1: Average all metrics across runs and export to CSV
    average_all_metrics(files_dir)

    # Step 2: Plot metrics for each individual run
    plot_single_runs(files_dir)

    # Step 3: Plot comparative metrics across all runs
    plot_cross_runs(files_dir, figures_dir)
