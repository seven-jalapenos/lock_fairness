
import csv
from pathlib import Path
from typing import Dict, Any, List
from .defs import Stats

class StatsExporter:
    """
    Handles exporting aggregated Stats from MetricAverager into formatted CSV files.
    """
    def __init__(self, output_dir: Path | str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(self, metrics_stats: Dict[str, Any]) -> None:
        """
        Iterates over the metrics dictionary and exports each metric to its own CSV.
        """
        # Group together simple scalar metrics so they can share a single CSV or be separate.
        # Given your prompt, we'll separate everything out cleanly.
        scalar_metrics: Dict[str, Stats] = {}

        for name, data in metrics_stats.items():
            if isinstance(data, Stats):
                scalar_metrics[name] = data
            elif isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], Stats):
                    self._export_1d_array(name, data)
                elif isinstance(data[0], list) and isinstance(data[0][0], Stats):
                    self._export_2d_matrix(name, data)
            else:
                print(f"Warning: Unknown or empty metric structure for '{name}'. Skipping.")

        if scalar_metrics:
            self._export_scalars(scalar_metrics)

    def _export_scalars(self, scalar_metrics: Dict[str, Stats]) -> None:
        """Writes all single-value metrics into a unified summary CSV."""
        filepath = self.output_dir / "summary_scalar_metrics.csv"
        
        with open(filepath, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Average', 'Standard_Deviation'])
            for name, stats in scalar_metrics.items():
                writer.writerow([name, stats.avg, stats.std])
        
        # print(f"Exported scalar metrics to: {filepath}")

    def _export_1d_array(self, metric_name: str, stats_list: List[Stats]) -> None:
        """Writes 1D thread-level array data (e.g., per_thread_wait_time) to a CSV."""
        filepath = self.output_dir / f"{metric_name}.csv"
        
        with open(filepath, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Thread_ID', 'Average', 'Standard_Deviation'])
            for thread_id, stats in enumerate(stats_list):
                writer.writerow([thread_id, stats.avg, stats.std])
                
        # print(f"Exported 1D metric '{metric_name}' to: {filepath}")

    def _export_2d_matrix(self, metric_name: str, stats_matrix: List[List[Stats]]) -> None:
        """
        Writes 2D matrix data (e.g., lock_transfer_matrix) to a CSV.
        Formats it as a flat table mapping Source -> Destination for easier analytical parsing.
        """
        filepath = self.output_dir / f"{metric_name}.csv"
        
        with open(filepath, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['From_Thread', 'To_Thread', 'Average', 'Standard_Deviation'])
            
            for i, row in enumerate(stats_matrix):
                for j, stats in enumerate(row):
                    writer.writerow([i, j, stats.avg, stats.std])
                    
        # print(f"Exported 2D matrix metric '{metric_name}' to: {filepath}")