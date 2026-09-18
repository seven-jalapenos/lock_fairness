

import argparse
from pathlib import Path

from analysis.log_analyzer import (WINDOW_SIZES_CYCLES, WINDOW_LABELS,
                                   WINDOW_COUNT_LABELS)
from analysis.single_run_plotter import SingleRunPlotter
from analysis.cross_run_plotter import CrossRunPlotter


# The headline set: throughput as the cost axis, Jain's index and the wait tail
# as the fairness axes, overtakes for order violation, self-transfer for
# convoying, and windowed Jain for the timescale at which fairness holds.
# Everything else stays in summary_scalar_metrics.csv as appendix material.
CROSS_RUN_METRICS = [
    'throughput_ops_per_Mcycle',
    'throughput_jain_index',
    'average_wait_time',
    'wait_p99',
    'wait_max',
    'overtake_percentage',
    'overtake_depth_p99_normalized',
    'self_transfer_rate',
    f'windowed_jain_{WINDOW_LABELS[10**6]}',
    # The 1e6 window holds fewer acquisitions than threads once a lock's
    # throughput collapses, which floors Jain at 1/n regardless of fairness;
    # 1e7 keeps the per-window sample count meaningful in that regime.
    f'windowed_jain_{WINDOW_LABELS[10**7]}',
    # Fixed-count companion to the two above. Both of those bin by wall time, so
    # once a lock collapses they hold fewer acquisitions than threads and report
    # sample count rather than fairness; this one holds k=10*threads either way.
    f'windowed_jain_{WINDOW_COUNT_LABELS[10]}',
    'total_CS_completions',
]


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

    Emits one figure set per (critical-section work size, pinning policy). Runs
    at different CS lengths aren't comparable on a single line, and neither are
    runs under different pinning policies -- pin 2 and 3 deliberately change
    which cores contend, so pooling them would draw two different experiments as
    one."""
    plotter = CrossRunPlotter(runs_csv_dir, figures_dir).load_data()

    if plotter.aggregated_data.empty:
        return

    data = plotter.aggregated_data

    # None is the group for directories predating the work dimension; it gets the
    # unsuffixed filenames the pipeline used to produce.
    work_values = sorted(data['work'].dropna().unique().tolist())
    if data['work'].isna().any():
        work_values.append(None)

    pin_values = sorted(data['pin'].dropna().unique().tolist())

    for work in work_values:
        work_suffix = '' if work is None else f'_w{int(work)}'
        for pin in pin_values:
            where = {'work': work, 'pin': pin}
            suffix = f"{work_suffix}_p{int(pin)}"

            # Most (work, pin) pairs in a sparse tree have no runs; skip them
            # rather than emitting a "not found" line per metric.
            mask = (data['pin'] == pin)
            mask &= data['work'].isna() if work is None else (data['work'] == work)
            if not mask.any():
                continue

            for metric in CROSS_RUN_METRICS:
                plotter.plot_metric(
                    metric,
                    x_axis='threads',
                    line_axis='lock_type',
                    save_csv=True,
                    where=where,
                    name_suffix=suffix
                )

            plotter.plot_timescale(where=where, name_suffix=suffix)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python -m scripts.plot_all',
        description='Render single-run and cross-run figures from the CSV tree.')
    parser.add_argument('--csv-dir', type=Path, default=Path('files/csv'),
                        help='root of the run directory tree (default files/csv)')
    parser.add_argument('--figures-dir', type=Path,
                        default=Path('files/final_figures'),
                        help='where cross-run figures are written')
    # Recomputing one scalar and redrawing the comparison figures shouldn't also
    # regenerate every per-run figure in the tree -- that is one directory of
    # plotting work per run in the whole sweep.
    parser.add_argument('--cross-only', action='store_true',
                        help='skip the per-run figures, draw only the cross-run '
                             'comparisons')
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.cross_only:
        plot_single_runs(args.csv_dir)
    plot_cross_runs(args.csv_dir, args.figures_dir)


if __name__ == '__main__':
    main()
