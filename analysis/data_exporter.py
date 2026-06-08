from .log_analyzer import LogAnalyzer
import pandas as pd
import os

class DataExporter:

    def __init__(self, data: pd.DataFrame, global_timeline: pd.DataFrame, csv_dir: str, run_name: str):
        self.data = data
        self.global_timeline = global_timeline
        self.csv_dir = csv_dir
        self.data_dir = os.path.join(csv_dir, 'data')
        self.timeline_dir = os.path.join(csv_dir, 'timeline')
        self.run_name = run_name

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.timeline_dir, exist_ok=True)

    def write_raw(self) -> None:
        """
        Writes the raw data and global timeline to parquet files.
        """
        if self.data is not None and self.global_timeline is not None:
            DataExporter.write_data(self.data, self.data_dir, self.run_name)
            DataExporter.write_global_timeline(self.global_timeline, self.timeline_dir, self.run_name)
        return

    def write_derived(self) -> None:
        """
        Writes derived data to CSV files.
        """
        pass

    @staticmethod
    def write_data(data: pd.DataFrame, output_dir: str, run_name: str) -> None:
        data.to_parquet(os.path.join(output_dir, f'{run_name}_data.parquet'), engine='pyarrow')

    @staticmethod
    def write_global_timeline(global_timeline: pd.DataFrame, output_dir: str, run_name: str) -> None:
        global_timeline.to_parquet(os.path.join(output_dir, f'{run_name}_timeline.parquet'), engine='pyarrow')
    
    def close(self) -> None:
        self.data = None
        self.global_timeline = None
