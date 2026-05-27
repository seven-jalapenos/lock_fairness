
from pathlib import Path

from analysis.metric_averager import MetricAverager
from analysis.stats_exporter import StatsExporter


def average_all_metrics(files_dir: Path) -> None:
    """
    Averages all metrics across all runs and exports them to CSV.
    """
    run_dirs = files_dir.glob("*")
    
    for run_dir in run_dirs:
        print(f"Processing {run_dir}...")
        averager = MetricAverager(run_dir).build_table()
        exporter = StatsExporter(run_dir)
        final_stats = averager.find_means_and_stds()
        exporter.export(final_stats)


if __name__ == '__main__':
    files_dir = Path("files/csv")
    average_all_metrics(files_dir)