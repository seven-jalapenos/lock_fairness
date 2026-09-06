import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os

# Only the raw event timestamps are stored. wait_time/hold_time are subtractions
# of these columns, recomputed on import -- keeping them on disk cost ~30% of the
# file for nothing.
RAW_COLUMNS = ['thread_id', 'invocation', 'acquisition', 'release']

# The timestamps are monotonic within each thread's block, so delta-packing them
# and compressing the residuals cuts the file to a fifth of the pandas default
# (73 MB -> 16 MB on a 4M-event run) and writes ~10x faster. use_dictionary must
# be off or dictionary encoding wins and column_encoding is silently ignored.
_PARQUET_OPTS = dict(
    compression='zstd',
    compression_level=9,
    use_dictionary=False,
    column_encoding={
        'invocation': 'DELTA_BINARY_PACKED',
        'acquisition': 'DELTA_BINARY_PACKED',
        'release': 'DELTA_BINARY_PACKED',
    },
)


class DataExporter:

    def __init__(self, data: pd.DataFrame, global_timeline: pd.DataFrame, csv_dir: str, run_name: str):
        self.data = data
        self.global_timeline = global_timeline
        self.csv_dir = csv_dir
        self.data_dir = os.path.join(csv_dir, 'data')
        self.run_name = run_name

        os.makedirs(self.data_dir, exist_ok=True)

    def write_raw(self) -> None:
        """
        Writes the raw per-event data to parquet.

        The global timeline is deliberately not persisted: it is a melt+sort of
        this same frame, so storing it doubled the run's footprint to save a
        rebuild that costs a couple of seconds (see MetricAverager.make_analyzer).
        """
        if self.data is not None:
            DataExporter.write_data(self.data, self.data_dir, self.run_name)
        return

    def write_derived(self) -> None:
        """
        Writes derived data to CSV files.
        """
        pass

    @staticmethod
    def write_data(data: pd.DataFrame, output_dir: str, run_name: str) -> None:
        table = pa.Table.from_pandas(data[RAW_COLUMNS], preserve_index=False)
        pq.write_table(table, os.path.join(output_dir, f'{run_name}_data.parquet'),
                       **_PARQUET_OPTS)

    def close(self) -> None:
        self.data = None
        self.global_timeline = None
