from pathlib import Path

from analysis.metric_averager import MetricAverager
from analysis.stats_exporter import StatsExporter


def average_all_metrics(files_dir: Path, only: set[str] | None = None) -> None:
    """
    Averages all metrics across all runs and exports them to CSV.

    `only` restricts processing to those run directory names. Averaging re-reads
    every per-iteration parquet and recomputes every metric, so a narrow sweep
    should not pay to redo the whole tree left behind by previous sweeps.

    Run directories are processed one at a time. This was briefly a process
    pool -- the directories are independent -- but the workers were killed or
    the stage hung, and a pool also had to buffer each directory's output to
    keep it from interleaving, which made a slow directory indistinguishable
    from a stuck one.
    """
    run_dirs = [
        d for d in sorted(files_dir.glob("*"))
        if d.is_dir() and (only is None or d.name in only)
    ]
    if not run_dirs:
        return

    failures: list[str] = []

    for index, run_dir in enumerate(run_dirs, start=1):
        # Announced before the work, not after, so the last line printed names
        # the directory currently being averaged.
        print(f"[{index}/{len(run_dirs)}] {run_dir.name}", flush=True)
        try:
            averager = MetricAverager(run_dir).build_table()
            exporter = StatsExporter(run_dir)
            final_stats = averager.find_means_and_stds()
            exporter.export(final_stats)
        except Exception as e:  # noqa: BLE001 - keep going, then fail loudly below
            print(f"[{index}/{len(run_dirs)}] FAILED {run_dir.name}: {e!r}",
                  flush=True)
            failures.append(f"{run_dir.name}: {e!r}")

    # One bad directory must not discard the rest of a multi-hour sweep's work --
    # every directory that succeeded has already written its CSVs -- but it must
    # not be reported as success either, or run_all sends a "completed" mail for
    # a tree with holes in it.
    if failures:
        print(f"\n{len(failures)} run director(ies) failed to average:", flush=True)
        for line in failures:
            print(f"  {line}", flush=True)
        raise RuntimeError(
            f"{len(failures)} of {len(run_dirs)} run directories failed to average; "
            "the rest were written. See the list above."
        )


if __name__ == '__main__':
    files_dir = Path("files/csv")
    average_all_metrics(files_dir)
