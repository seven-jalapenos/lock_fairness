import argparse
from pathlib import Path
from typing import Iterable, Optional

from analysis.metric_averager import MetricAverager, RECOMPUTABLE_SCALARS
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


def update_all_metrics(files_dir: Path, names: Optional[Iterable[str]] = None,
                       only: set[str] | None = None) -> None:
    """Recompute just `names` for each run directory and merge them into its
    existing summary CSV.

    The full averaging pass re-reads every parquet, rebuilds each global timeline
    and loads or re-runs the overtake scan, which is the expensive part of a
    sweep. Adding one cheap scalar to a tree that has already been analyzed
    shouldn't pay any of that, so this walks the same directories but computes
    only what was asked for and merges rather than rewrites.
    """
    run_dirs = [
        d for d in sorted(files_dir.glob("*"))
        if d.is_dir() and (only is None or d.name in only)
    ]
    if not run_dirs:
        return

    failures: list[str] = []

    for index, run_dir in enumerate(run_dirs, start=1):
        print(f"[{index}/{len(run_dirs)}] {run_dir.name}", flush=True)
        try:
            updates = MetricAverager(run_dir).recompute_scalars(names)
            StatsExporter(run_dir).update_scalars(updates)
        except Exception as e:  # noqa: BLE001 - keep going, then fail loudly below
            print(f"[{index}/{len(run_dirs)}] FAILED {run_dir.name}: {e!r}",
                  flush=True)
            failures.append(f"{run_dir.name}: {e!r}")

    # Same contract as average_all_metrics: a directory that failed leaves its
    # previous values untouched, the rest are written, and the caller still hears
    # about it rather than getting a silent partial result.
    if failures:
        print(f"\n{len(failures)} run director(ies) failed to update:", flush=True)
        for line in failures:
            print(f"  {line}", flush=True)
        raise RuntimeError(
            f"{len(failures)} of {len(run_dirs)} run directories failed to update; "
            "the rest were written. See the list above."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python -m scripts.all_metrics',
        description='Average per-run metrics into summary_scalar_metrics.csv.')
    parser.add_argument('--csv-dir', type=Path, default=Path('files/csv'),
                        help='root of the run directory tree (default files/csv)')
    parser.add_argument('--only', nargs='+', metavar='RUN_DIR', default=None,
                        help='restrict to these run directory names')
    # nargs='*' so three cases stay distinguishable: flag absent is the full
    # averaging pass (unchanged default), flag bare updates every recomputable
    # scalar, flag with names updates just those.
    parser.add_argument('--metrics', nargs='*', metavar='NAME', default=None,
                        choices=sorted(RECOMPUTABLE_SCALARS),
                        help='recompute only these scalars and merge them into '
                             'the existing summary CSV, skipping the full pass. '
                             f'Choices: {", ".join(sorted(RECOMPUTABLE_SCALARS))}')
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    only = set(args.only) if args.only else None
    if args.metrics is not None:
        update_all_metrics(args.csv_dir, names=(args.metrics or None), only=only)
    else:
        average_all_metrics(args.csv_dir, only=only)


if __name__ == '__main__':
    main()
