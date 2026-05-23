
from .data_importer import import_parquet
from .log_analyzer import LogAnalyzer

from pathlib import Path
from collections import namedtuple
from statistics import mean, stdev
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .defs import Stats

data_dir = 'data'
timeline_dir = 'timeline'

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

        self.avg_metrics: Dict[str, Any] = {
            'per_thread_wait_time': [],
            'percent_time_in_CS': Stats, 
            'total_CS_completions': Stats,
            'lock_transfer_matrix': [],
            'overtake_percentage': Stats,
            'average_overtake_depth': Stats,
            'rank_inversion_penalty': Stats
        }

    def build_table(self) -> MetricAverager:
        self.all_metrics, self.metric_vars, self.thread_count = self.all_metrics_and_thread_count(
            self.data_dir, self.timeline_dir
        )
        return self

    def make_analyzer(self, data_file: Path, timeline_file: Path) -> LogAnalyzer:
        data = import_parquet(data_file)
        timeline = import_parquet(timeline_file)
        return LogAnalyzer(data, timeline)
    
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
            analyzer = self.make_analyzer(data_file, timeline_file)
            if not threads:
                threads = analyzer.num_threads

            # 1. Fetch values
            avg_wait, var_wait = analyzer.find_avg_per_thread_wait_time()
            avg_overtake, var_overtake = analyzer.overtake_percentage()
            avg_depth, var_depth = analyzer.average_overtake_depth()

            # 2. Store run metrics in a dictionary (much safer for arrays than 1-row DataFrames)
            run_records.append({
                'run': run,
                'operation_count': analyzer.operation_count,
                'event_count': analyzer.event_count,
                'per_thread_wait_time': avg_wait,
                # 'percent_time_in_CS': analyzer.percent_time_in_CS(),
                'overtake_percentage': avg_overtake,
                'average_overtake_depth': avg_depth,
                'total_CS_completions': analyzer.total_CS_completions(),
                'lock_transfer_matrix': analyzer.lock_transfer_matrix(),
                'rank_inversion_penalty': analyzer.rank_inversion_penalty()
            })

            # 3. Store variance metrics
            var_records.append({
                'run': run,
                'per_thread_wait_time': var_wait,
                'overtake_percentage': var_overtake,
                'average_overtake_depth': var_depth,
            })
        
        # Convert list of dicts to DataFrame in one shot
        return pd.DataFrame(run_records), pd.DataFrame(var_records), threads

    def single_mean_stats(self, N: np.ndarray, mu: np.ndarray, var: np.ndarray) -> Any: # Returns Stats
        """
        Calculate statistics for a set of weighted means (Pooled Mean & Variance).
        N: array of counts
        mu: array of means
        var: array of variances
        """
        N_total = np.sum(N)
        mu_total = np.dot(N, mu) / N_total

        # Law of Total Variance
        squared_mean_drift = (mu - mu_total) ** 2
        inner_component = var + squared_mean_drift
        var_total = np.dot(N, inner_component) / N_total

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
        avg = val_total / N_total

        squared_drift = (values - avg) ** 2
        var_total = np.dot(N, squared_drift) / N_total
        std_total = np.sqrt(var_total)

        return Stats(avg=avg, std=std_total)
    
    def find_means_and_stds(self) -> Dict[str, Any]:
        """
        Averages each metric across runs, returning a dict of Stats
        """
        metrics_stats = {}
        # Make sure 'operation_count' is in self.all_metrics, else default to an array of 1s
        operation_counts = self.all_metrics['operation_count'].to_numpy()

        # 1. Single Value Mean Metrics
        for metric in ['overtake_percentage', 'average_overtake_depth']:
            vals = self.all_metrics[metric].to_numpy()
            vars_ = self.metric_vars[metric].to_numpy()
            metrics_stats[metric] = self.single_mean_stats(operation_counts, vals, vars_)
        
        # 2. Array Mean Metrics (1D Arrays per run)
        metric = 'per_thread_wait_time'
        # Stack converts Series of arrays into a 2D numpy matrix: shape (num_runs, num_threads)
        stacked_vals = np.stack(self.all_metrics[metric].to_list())
        stacked_vars = np.stack(self.metric_vars[metric].to_list())
        
        thread_stats = []
        for i in range(self.thread_count):
            # Extract the i-th thread across all runs
            thread_metric_values = stacked_vals[:, i]
            thread_metric_vars = stacked_vars[:, i]
            
            stat = self.single_mean_stats(operation_counts, thread_metric_values, thread_metric_vars)
            thread_stats.append(stat)
            
        metrics_stats[metric] = thread_stats

        # 3. Single Value Accumulated Metrics
        for metric in ['total_CS_completions', 'rank_inversion_penalty']:
            if metric in self.all_metrics:
                vals = self.all_metrics[metric].to_numpy()
                metrics_stats[metric] = self.single_acc_stats(operation_counts, vals)

        # 4. Matrix Accumulated Metrics (2D Arrays per run)
        metric = 'lock_transfer_matrix'
        if metric in self.all_metrics:
            # Shape: (num_runs, num_threads, num_threads)
            stacked_matrices = np.stack(self.all_metrics[metric].to_list()) 
            matrix_stats = []
            
            for i in range(self.thread_count):
                row_stats = []
                for j in range(self.thread_count):
                    # Extract the (i, j) relationship across all runs
                    vals_ij = stacked_matrices[:, i, j]
                    stat = self.single_acc_stats(operation_counts, vals_ij)
                    row_stats.append(stat)
                matrix_stats.append(row_stats)
                
            metrics_stats[metric] = matrix_stats

        return metrics_stats

