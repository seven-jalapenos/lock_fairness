from .data_importer import import_parquet
from .log_analyzer import (LogAnalyzer, COVERAGE_WARN_THRESHOLD,
                           WINDOW_SIZES_CYCLES, WINDOW_LABELS)
from .defs import Stats

from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any

data_dir = 'data'
timeline_dir = 'timeline'

# Metrics that are already a per-run summary (a percentile, a fairness index, a
# ratio). There is no within-run variance to combine for these, so they are
# averaged plainly across reps rather than pooled -- see simple_stats.
SIMPLE_SCALARS = [
    'throughput_ops_per_Mcycle',
    'throughput_jain_index',
    'throughput_ratio',
    'wait_time_cov',
    'mean_hold_time',
    'log_coverage',
    'ordering_ambiguity_fraction',
    'self_transfer_rate',
    'transfer_entropy',
    'average_overtake_depth_normalized',
    'wait_p50', 'wait_p90', 'wait_p99', 'wait_p999', 'wait_max',
    'overtake_depth_p99', 'overtake_depth_max',
    'overtake_depth_p99_normalized', 'overtake_depth_max_normalized',
] + [f'windowed_jain_{WINDOW_LABELS[w]}' for w in WINDOW_SIZES_CYCLES]


class MetricAverager:
    """
    MetricAverager is responsible for averaging metrics accross each parameter's 10 trials
    """
    def __init__(self, run_dir: Path):
        self.run_dir: Path = run_dir
        self.data_dir: Path = run_dir / data_dir
        self.timeline_dir: Path = run_dir / timeline_dir

        self.all_metrics = pd.DataFrame()
        self.metric_vars = pd.DataFrame()
        self.thread_count: int = 0

    def build_table(self) -> 'MetricAverager':
        self.all_metrics, self.metric_vars, self.thread_count = self.all_metrics_and_thread_count(
            self.data_dir, self.timeline_dir
        )
        return self

    def make_analyzer(self, data_file: Path, timeline_file: Path, overtake_file: Path) -> LogAnalyzer:
        data = import_parquet(data_file)
        timeline = import_parquet(timeline_file)

        if overtake_file is not None and overtake_file.exists():
            overtake = import_parquet(overtake_file)
            # A cache written before ambiguous_acquisitions existed is stale.
            # _run_complete only validates the data/timeline parquet, so a resumed
            # sweep would otherwise silently reuse the old schema and report NaN
            # ambiguity forever.
            if 'ambiguous_acquisitions' in overtake.columns:
                return LogAnalyzer(data, timeline, overtake)

        la = LogAnalyzer(data, timeline)
        la.print_overtake(overtake_file)
        return la

    def all_metrics_and_thread_count(self, data_dir: Path, timeline_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, int]:
        """
        Returns DataFrame with metrics from every run
        """
        data_files = sorted(data_dir.glob('*.parquet'))
        timeline_files = sorted(timeline_dir.glob('*.parquet'))

        run_records = []
        var_records = []
        threads = 0

        for run, (data_file, timeline_file) in enumerate(zip(data_files, timeline_files)):
            overtake_file = self.run_dir / 'overtake' / timeline_file.name
            analyzer = self.make_analyzer(data_file, timeline_file, overtake_file)
            if not threads:
                threads = analyzer.num_threads

            # 1. Fetch values
            avg_wait, var_wait, count_wait = analyzer.find_avg_per_thread_wait_time()
            avg_overtake, var_overtake = analyzer.overtake_percentage()
            avg_depth, var_depth = analyzer.average_overtake_depth()
            transfer = analyzer.lock_transfer_matrix()
            coverage = analyzer.log_coverage()

            # A truncated run understates unfairness rather than merely losing
            # samples: the threads a lock favors fill their buffers first and go
            # dark, so the survivors look more equal than they were. Flag it here,
            # where the run it belongs to is still named.
            if np.isfinite(coverage) and coverage < COVERAGE_WARN_THRESHOLD:
                print(
                    f"  WARNING: {self.run_dir.name} iteration {run} has "
                    f"log_coverage={coverage:.4f} "
                    f"(<{COVERAGE_WARN_THRESHOLD}) -- the log truncated and "
                    "every count-based metric from it is biased. Raise "
                    "LOG_BUDGET_BYTES or shorten DURATION and re-run.",
                    flush=True
                )

            denom = max(1, int(analyzer.num_threads) - 1)

            record = {
                'run': run,
                'operation_count': analyzer.operation_count,
                'per_thread_wait_time': avg_wait,
                'per_thread_wait_count': count_wait,
                'per_thread_throughput': analyzer.per_thread_throughput(),
                'overtake_percentage': avg_overtake,
                'average_overtake_depth': avg_depth,
                'average_overtake_depth_normalized': avg_depth / denom,
                'total_CS_completions': analyzer.total_CS_completions(),
                'lock_transfer_matrix': transfer,
                'throughput_ops_per_Mcycle': analyzer.throughput_ops_per_Mcycle(),
                'throughput_jain_index': analyzer.throughput_jain_index(),
                'throughput_ratio': analyzer.throughput_ratio(),
                'wait_time_cov': analyzer.wait_time_cov(),
                'mean_hold_time': analyzer.mean_hold_time(),
                'log_coverage': coverage,
                'ordering_ambiguity_fraction': analyzer.ordering_ambiguity_fraction(),
                'self_transfer_rate': analyzer.self_transfer_rate(transfer),
                'transfer_entropy': analyzer.transfer_entropy(transfer),
            }
            record.update(analyzer.wait_time_percentiles())
            record.update(analyzer.overtake_depth_percentiles())
            for window in WINDOW_SIZES_CYCLES:
                record[f'windowed_jain_{WINDOW_LABELS[window]}'] = analyzer.windowed_jain(window)

            # 2. Store run metrics in a dictionary (much safer for arrays than 1-row DataFrames)
            run_records.append(record)

            # 3. Store variance metrics
            var_records.append({
                'run': run,
                'per_thread_wait_time': var_wait,
                'overtake_percentage': var_overtake,
                'average_overtake_depth': var_depth,
            })

            analyzer.close()

        # Convert list of dicts to DataFrame in one shot
        return pd.DataFrame(run_records), pd.DataFrame(var_records), threads

    def single_mean_stats(self, N: np.ndarray, mu: np.ndarray, var: np.ndarray) -> Any: # Returns Stats
        """
        Calculate statistics for a set of weighted means (Unbiased Pooled Mean & Variance).
        """
        N_total = np.sum(N)
        mu_total = np.dot(N, mu) / N_total

        # Explicitly calculate Sum of Squares (Within-group + Between-group)
        ss_within = np.dot(N - 1, var)
        ss_between = np.dot(N, (mu - mu_total) ** 2)

        # Divide by total degrees of freedom for an unbiased sample variance
        var_total = (ss_within + ss_between) / (N_total - 1)
        std_total = np.sqrt(var_total)

        return Stats(avg=mu_total, std=std_total)

    def single_acc_stats(self, N: np.ndarray, values: np.ndarray) -> Any: # Returns Stats
        """
        Calculate statistics for a set of accumulated values across runs.
        N: array of counts
        values: array of values
        """
        N_total = np.sum(N)
        val_total = np.sum(values)

        # Average per operation
        avg_rate = val_total / N_total
        run_rates = values / N

        squared_drift = (run_rates - avg_rate) ** 2
        var_total = np.dot(N, squared_drift) / N_total
        std_total = np.sqrt(var_total)

        return Stats(avg=avg_rate, std=std_total)

    @staticmethod
    def simple_stats(values: np.ndarray) -> Stats:
        """
        Mean and standard deviation *across reps* for metrics that are already a
        per-run summary.

        The pooled-variance formulas in single_mean_stats combine a within-group
        variance with a between-group one, which needs each run to contribute a
        mean over N samples. A percentile or a fairness index isn't a mean of
        anything -- there is no within-run variance to pool -- so the only honest
        spread is the run-to-run one.
        """
        vals = np.asarray(values, dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return Stats(avg=float('nan'), std=float('nan'))
        if vals.size == 1:
            return Stats(avg=float(vals[0]), std=0.0)
        return Stats(avg=float(vals.mean()), std=float(vals.std(ddof=1)))

    def find_means_and_stds(self) -> Dict[str, Any]:
        """
        Averages each metric across runs, returning a dict of Stats
        """
        metrics_stats: Dict[str, Any] = {}
        if self.all_metrics.empty:
            return metrics_stats

        # Make sure 'operation_count' is in self.all_metrics, else default to an array of 1s
        operation_counts = self.all_metrics['operation_count'].to_numpy()

        # 1. Single Value Mean Metrics (genuine per-run means, so pool properly)
        for metric in ['overtake_percentage', 'average_overtake_depth']:
            vals = self.all_metrics[metric].to_numpy()
            vars_ = self.metric_vars[metric].to_numpy()
            metrics_stats[metric] = self.single_mean_stats(operation_counts, vals, vars_)

        # 1b. Per-run summary scalars: averaged across reps, not pooled.
        for metric in SIMPLE_SCALARS:
            if metric in self.all_metrics:
                metrics_stats[metric] = self.simple_stats(self.all_metrics[metric].to_numpy())

        # 2. Array Mean Metrics (1D Arrays per run)
        metric = 'per_thread_wait_time'
        # Stack converts Series of arrays into a 2D numpy matrix: shape (num_runs, num_threads)
        stacked_vals = np.stack(self.all_metrics[metric].to_list())
        stacked_vars = np.stack(self.metric_vars[metric].to_list())
        stacked_counts = np.stack(self.all_metrics['per_thread_wait_count'].to_list())

        overall_stat = self.single_mean_stats(
            stacked_counts.flatten(),
            stacked_vals.flatten(),
            stacked_vars.flatten()
        )
        metrics_stats['average_wait_time'] = overall_stat

        thread_stats = []
        for i in range(self.thread_count):
            # Extract the i-th thread across all runs
            thread_metric_counts = stacked_counts[:, i]
            thread_metric_values = stacked_vals[:, i]
            thread_metric_vars = stacked_vars[:, i]

            stat = self.single_mean_stats(thread_metric_counts, thread_metric_values, thread_metric_vars)
            thread_stats.append(stat)

        metrics_stats[metric] = thread_stats

        # 2b. Per-thread completion counts: a plain count per run, so across-rep
        # mean/std is the whole story.
        if 'per_thread_throughput' in self.all_metrics:
            stacked_tput = np.stack(self.all_metrics['per_thread_throughput'].to_list())
            metrics_stats['per_thread_throughput'] = [
                self.simple_stats(stacked_tput[:, i]) for i in range(self.thread_count)
            ]

        # 3. Single Value Accumulated Metrics
        for metric in ['total_CS_completions']:
            if metric in self.all_metrics:
                vals = self.all_metrics[metric].to_numpy()
                metrics_stats[metric] = self.single_acc_stats(np.ones_like(vals), vals)

        # 4. Matrix Accumulated Metrics (2D Arrays per run)
        metric = 'lock_transfer_matrix'
        if metric in self.all_metrics:
            # Shape: (num_runs, num_threads, num_threads)
            stacked_matrices = np.stack(self.all_metrics[metric].to_list()).astype(np.float64)

            # Normalize each run to handoff *probabilities* before averaging, so
            # reps of differing length contribute equally, then take the mean and
            # spread across reps in one vectorized pass. The previous version
            # looped single_acc_stats over all n^2 cells in Python, which cost
            # minutes at 28 threads to produce the same heatmap.
            totals = stacked_matrices.sum(axis=(1, 2), keepdims=True)
            totals[totals == 0] = 1.0
            probs = stacked_matrices / totals

            avg = probs.mean(axis=0)
            std = probs.std(axis=0, ddof=1) if probs.shape[0] > 1 else np.zeros_like(avg)

            metrics_stats[metric] = [
                [Stats(avg=float(avg[i, j]), std=float(std[i, j]))
                 for j in range(self.thread_count)]
                for i in range(self.thread_count)
            ]

        return metrics_stats
