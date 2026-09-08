import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from analysis.metric_averager import MetricAverager
from analysis.stats_exporter import StatsExporter

# Each worker holds a whole run's frames at once -- roughly 0.5 GB per million
# critical-section completions, so ~8 GB at work=10000 and several times that at
# short critical sections, where a run turns over far more operations. Memory,
# not cores, is what bounds this, hence the cap well below a big machine's core
# count.
DEFAULT_MAX_WORKERS = 8


def _process_run_dir(run_dir: Path) -> tuple[str, list[str]]:
    """Average one run directory's reps and export the CSVs.

    Returns the messages produced rather than printing them, so a pool of these
    doesn't interleave one run's coverage warnings into another's output.
    """
    import io
    import contextlib

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        averager = MetricAverager(run_dir).build_table()
        exporter = StatsExporter(run_dir)
        final_stats = averager.find_means_and_stds()
        exporter.export(final_stats)
    return run_dir.name, [line for line in buffer.getvalue().splitlines() if line.strip()]


def average_all_metrics(files_dir: Path, only: set[str] | None = None,
                        workers: int | None = None) -> None:
    """
    Averages all metrics across all runs and exports them to CSV.

    `only` restricts processing to those run directory names. Averaging re-reads
    every per-iteration parquet and recomputes every metric, so a narrow sweep
    should not pay to redo the whole tree left behind by previous sweeps.

    Run directories are independent of each other, so they are processed in
    parallel; this stage used to be the one single-threaded phase of a sweep,
    with every core but one idle. `workers` caps the pool -- see
    DEFAULT_MAX_WORKERS for why the ceiling is about memory rather than cores.

    NOTE: Python 3.14 starts subprocesses with `forkserver` on Linux, so each
    worker re-imports the __main__ module. Whatever calls this must therefore be
    import-safe -- guarded behind `if __name__ == '__main__':`, as run_all.py and
    this module are. A caller that runs a sweep at import time will run it again
    in every worker.
    """
    run_dirs = [
        d for d in sorted(files_dir.glob("*"))
        if d.is_dir() and (only is None or d.name in only)
    ]
    if not run_dirs:
        return

    if workers is None or workers <= 0:
        workers = min(os.cpu_count() or 1, DEFAULT_MAX_WORKERS)
    workers = max(1, min(workers, len(run_dirs)))

    failures: list[str] = []

    def report(index: int, run_dir: Path, result, error) -> None:
        if error is not None:
            print(f"[{index}/{len(run_dirs)}] FAILED {run_dir.name}: {error!r}",
                  flush=True)
            failures.append(f"{run_dir.name}: {error!r}")
            return
        name, messages = result
        print(f"[{index}/{len(run_dirs)}] {name}", flush=True)
        for line in messages:
            print(line, flush=True)

    if workers == 1:
        for index, run_dir in enumerate(run_dirs, start=1):
            try:
                report(index, run_dir, _process_run_dir(run_dir), None)
            except Exception as e:  # noqa: BLE001 - keep going, then fail loudly below
                report(index, run_dir, None, e)
    else:
        print(f"Averaging {len(run_dirs)} run directories across {workers} workers...",
              flush=True)
        # max_tasks_per_child so a worker is replaced periodically rather than
        # accumulating the peak RSS of every run directory it has touched -- the
        # same reason scripts/runner.py forces a collection between runs.
        with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=4) as pool:
            futures = {pool.submit(_process_run_dir, d): d for d in run_dirs}
            # Reported as they finish rather than in submission order, so a slow
            # run directory doesn't hold back progress on everything behind it.
            for index, future in enumerate(as_completed(futures), start=1):
                try:
                    report(index, futures[future], future.result(), None)
                except Exception as e:  # noqa: BLE001 - see below
                    report(index, futures[future], None, e)

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
