        

from pathlib import Path

from analysis.single_run_plotter import SingleRunPlotter
from analysis.cross_run_plotter import CrossRunPlotter


def plot_single_runs(run_dir: Path) -> None:
    """Utility function to plot all metrics from a single run."""
    run_dirs = list(run_dir.glob("*"))
    for dir in run_dirs:
        print(f"Plotting {dir}...")
        plotter = SingleRunPlotter(dir, dir / "figures")
        plotter.plot_all()

def plot_cross_runs(runs_csv_dir: Path, figures_dir: Path) -> None:
    """Utility function to plot comparative metrics across multiple runs."""
    metrics = [
        'overtake_percentage',
        'average_overtake_depth',
        'rank_inversion_penalty',
        'total_CS_completions'
        ]
    plotter = CrossRunPlotter(runs_csv_dir, figures_dir).load_data()

    for metric in metrics:
        plotter.plot_metric(metric, x_axis='threads', line_axis='lock_type')
    

if __name__ == '__main__':
    runs_csv_dir = Path("files/csv")
    figures_dir = Path("files/final_figures")
    plot_single_runs(runs_csv_dir)
    plot_cross_runs(runs_csv_dir, figures_dir)
