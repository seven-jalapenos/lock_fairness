from .log_analyzer import LogAnalyzer
import pandas as pd
import os

class DataExporter:

    def __init__(self, log_analyzer: LogAnalyzer, csv_dir: str, run_name: str):
        self.log_analyzer = log_analyzer
        self.csv_dir = csv_dir
        self.data_dir = os.path.join(csv_dir, 'data')
        self.timeline_dir = os.path.join(csv_dir, 'timeline')
        self.run_name = run_name

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.timeline_dir, exist_ok=True)

    def write_raw(self) -> None:
        """
        Writes the raw data and global timeline to CSV files.
        """
        DataExporter.write_data(self.log_analyzer, self.data_dir, self.run_name)
        DataExporter.write_global_timeline(self.log_analyzer, self.timeline_dir, self.run_name)
    
    def write_derived(self) -> None:
        """
        Writes derived data to CSV files.
        """
        pass

    @staticmethod
    def write_data(log_analyzer: LogAnalyzer, output_dir: str, run_name: str) -> None:
        log_analyzer._data.to_parquet(os.path.join(output_dir, f'{run_name}_data.parquet'), engine='pyarrow')

    @staticmethod
    def write_global_timeline(log_analyzer: LogAnalyzer, output_dir: str, run_name: str) -> None:
        log_analyzer._global_timeline.to_parquet(os.path.join(output_dir, f'{run_name}_timeline.parquet'), engine='pyarrow')
