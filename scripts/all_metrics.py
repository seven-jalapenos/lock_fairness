
from pathlib import Path

from analysis.metric_averager import MetricAverager
from analysis.stats_exporter import StatsExporter


def average_all_metrics(files_dir: Path, only: set[str] | None = None) -> None:
    """
    Averages all metrics across all runs and exports them to CSV.

    `only` restricts processing to those run directory names. Averaging re-reads
    every per-iteration parquet and recomputes every metric, so a narrow sweep
    should not pay to redo the whole tree left behind by previous sweeps.
    """
    run_dirs = files_dir.glob("*")
    
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue
        if only is not None and run_dir.name not in only:
            continue
        print(f"Processing {run_dir}...")
        averager = MetricAverager(run_dir).build_table()
        exporter = StatsExporter(run_dir)
        final_stats = averager.find_means_and_stds()
        exporter.export(final_stats)


if __name__ == '__main__':
    files_dir = Path("files/csv")
    average_all_metrics(files_dir)