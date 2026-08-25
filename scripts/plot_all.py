        

from pathlib import Path

from analysis.single_run_plotter import SingleRunPlotter
from analysis.cross_run_plotter import CrossRunPlotter


def plot_single_runs(run_dir: Path, only: set[str] | None = None) -> None:
    """Utility function to plot all metrics from a single run.

    `only` restricts plotting to those run directory names, so a narrow sweep
    doesn't regenerate figures for every directory from earlier sweeps."""
    run_dirs = list(run_dir.glob("*"))
    for dir in run_dirs:
        if not dir.is_dir():
            continue
        if only is not None and dir.name not in only:
            continue
        print(f"Plotting {dir}...")
        plotter = SingleRunPlotter(dir, dir / "figures")
        plotter.plot_all()

def plot_cross_runs(runs_csv_dir: Path, figures_dir: Path) -> None:
    """Utility function to plot comparative metrics across multiple runs.

    Deliberately unscoped: these are the comparison figures, so a sweep of one
    lock should still be drawn against everything previously measured.

    Emits one figure set per critical-section work size. Runs at different CS
    lengths aren't comparable on a single line -- pooling them would put two
    unrelated points at the same thread count."""
    metrics = [
        'overtake_percentage',
        'average_overtake_depth',
        'rank_inversion_penalty',
        'total_CS_completions',
        'average_wait_time'
        ]
    plotter = CrossRunPlotter(runs_csv_dir, figures_dir).load_data()

    if plotter.aggregated_data.empty:
        return

    # None is the group for directories predating the work dimension; it gets the
    # unsuffixed filenames the pipeline used to produce.
    work_values = sorted(
        plotter.aggregated_data['work'].dropna().unique().tolist()
    )
    if plotter.aggregated_data['work'].isna().any():
        work_values.append(None)

    for work in work_values:
        suffix = '' if work is None else f'_w{int(work)}'
        for metric in metrics:
            plotter.plot_metric(
                metric,
                x_axis='threads',
                line_axis='lock_type',
                save_csv=True,
                where={'work': work},
                name_suffix=suffix
            )

if __name__ == '__main__':
    runs_csv_dir = Path("files/csv")
    figures_dir = Path("files/final_figures")
    plot_single_runs(runs_csv_dir)
    plot_cross_runs(runs_csv_dir, figures_dir)
