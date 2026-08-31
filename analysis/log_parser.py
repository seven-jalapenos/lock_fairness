
import numpy as np
import pandas as pd
import os
from .parse_offsets import parse_tsc_offsets
from .defs import EVENT_CODES

def core_for_thread(tid: int, num_threads: int, num_cores: int,
                    policy: int) -> int | None:
    """Map a thread id to the core it was pinned to, or None if it wasn't pinned.

    MUST mirror the core_ids assignment in src/runner/main.cpp. TSC calibration
    subtracts a *per-core* offset, so if the two disagree about which core a
    thread ran on, every cross-thread timestamp comparison -- and therefore every
    overtake -- is silently wrong rather than merely noisy."""
    if policy == 1:
        return tid % num_cores

    hot_core = num_cores // 2

    if policy == 2:
        return hot_core

    if policy == 3:
        half = num_threads // 2
        if tid < half:
            return hot_core
        if num_cores <= 1:
            return 0
        # Cold half skips the hot core, matching main.cpp.
        slot = (tid - half) % (num_cores - 1)
        return slot if slot < hot_core else slot + 1

    # policy 0: unpinned. Threads migrate, so there is no single core whose
    # offset applies to the run -- calibration is undefined, not just skipped.
    return None


class LogParser:
    def __init__(self, file_path: str, offset_file_path: str, pinning_policy: int = 1):
        self.file_path = file_path
        self.offset_file_path = offset_file_path
        self.pinning_policy = pinning_policy

        self.offset_table = parse_tsc_offsets(offset_file_path)

        self.all_threads_data = self.parse_logs()

    def close(self) -> None:
        self.all_threads_data = None
        self.offset_table = None

    def _read_blocks(self, f, log_entry_dtype, size_t_dtype, file_size) -> list:
        """Read every per-thread block from an open log, in thread-spawn order."""
        blocks = []
        while True:
            # 1. Read the number of entries (size_t)
            size_data = np.fromfile(f, dtype=size_t_dtype, count=1)
            if size_data.size == 0:
                break

            num_entries = int(size_data[0])

            # Guard against a corrupt/truncated log: np.fromfile preallocates
            # `count` elements up front, so a garbage num_entries (e.g. from a
            # short write) would try to allocate terabytes and hang/thrash the
            # box instead of failing. A thread can't have more entries than fit
            # in the remaining bytes.
            max_entries = (file_size - f.tell()) // log_entry_dtype.itemsize
            if num_entries > max_entries:
                raise ValueError(
                    f"Corrupt log {self.file_path}: thread {len(blocks)} claims "
                    f"{num_entries} entries but only {max_entries} fit in the "
                    f"remaining file. Log is likely truncated or malformed."
                )

            # 2. Read all LogEntries for this thread in one block
            blocks.append(np.fromfile(f, dtype=log_entry_dtype, count=num_entries))
        return blocks

    def parse_logs(self) -> pd.DataFrame:
        # Define the LogEntry structure (3 x uint64_t)
        log_entry_dtype = np.dtype([
            ('invocation', 'u8'),
            ('acquisition', 'u8'),
            ('release', 'u8')
        ])

        size_t_dtype = np.dtype('u8')

        if not os.path.exists(self.file_path):
            print(f"Error: {self.file_path} not found.")
            return pd.DataFrame() # Return empty DataFrame instead of None

        file_size = os.path.getsize(self.file_path)

        # Read every block before calibrating: pinning policy 3 needs the total
        # thread count to know which half of the threads sat on the hot core, and
        # taking it from the file rather than from a caller-supplied argument
        # makes it impossible for the two to disagree. Calibration is in-place on
        # the arrays already read, so this costs no extra memory.
        with open(self.file_path, 'rb') as f:
            blocks = self._read_blocks(f, log_entry_dtype, size_t_dtype, file_size)

        if not blocks:
            return pd.DataFrame()

        num_threads = len(blocks)
        num_cores = 0 if self.offset_table is None else len(self.offset_table)

        thread_dfs = [] # We will store a flat DataFrame for each thread here
        for thread_id, data in enumerate(blocks):
            self.calibrate_log(data, thread_id, num_threads, num_cores)

            # --- THE FLATTENING MAGIC HAPPENS HERE ---
            # Passing the structured array directly to Pandas splits
            # 'invocation', 'acquisition', and 'release' into individual flat columns!
            df_thread = pd.DataFrame(data)

            # Add the thread_id as a standard column (Pandas broadcasts the scalar automatically)
            df_thread.insert(0, 'thread_id', thread_id)

            # Calculate metrics as flat columns using clean vectorized math
            df_thread['wait_time'] = df_thread['acquisition'] - df_thread['invocation']
            df_thread['hold_time'] = df_thread['release'] - df_thread['acquisition']

            thread_dfs.append(df_thread)

        # Vertically stack all individual thread dataframes into one massive table
        final_df = pd.concat(thread_dfs, ignore_index=True)
        return final_df

    def calibrate_log(self, data: np.ndarray, thread_id: int,
                      num_threads: int, num_cores: int) -> np.ndarray:
        """
        Applies offsets to make TSC values comparable across threads.
        Mutates `data` in place and returns it.
        """
        if self.offset_table is None or num_cores == 0:
            return data

        core = core_for_thread(thread_id, num_threads, num_cores, self.pinning_policy)

        if core is None:
            # Warn once per parser, not once per thread.
            if not getattr(self, '_warned_unpinned', False):
                self._warned_unpinned = True
                print(
                    f"Warning: {self.file_path} was run unpinned (policy "
                    f"{self.pinning_policy}); threads migrate between cores, so no "
                    "per-core TSC offset applies. Timestamps are left uncalibrated "
                    "and cross-thread ordering carries unquantified error."
                )
            return data

        if core >= num_cores:
            raise ValueError(
                f"{self.offset_file_path} has offsets for {num_cores} cores but "
                f"pinning policy {self.pinning_policy} puts thread {thread_id} of "
                f"{num_threads} on core {core}. The offsets file is from a machine "
                "with a different core count -- re-run the benchmark to regenerate it."
            )

        offset = int(self.offset_table[core])
        if offset == 0:
            return data

        # Timestamps are uint64, so apply the offset as modular arithmetic: the
        # two's-complement form of a negative offset subtracts to the same value
        # a signed subtraction would, and every consumer looks at differences.
        delta = np.uint64(offset & 0xFFFFFFFFFFFFFFFF)

        # Vectorized subtraction on the entire NumPy structured array
        data['invocation']  -= delta
        data['acquisition'] -= delta
        data['release']     -= delta

        return data


#########################
#
#     Global Timeline
#
#########################

def create_global_timeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a flat lock event DataFrame into a chronological global timeline.
    """
    if df.empty:
        return pd.DataFrame()
    # 1. 'Melt' the three timestamp columns into a single chronological column
    timeline_df = df.melt(
        id_vars=['thread_id'],                                # Keep thread_id as is
        value_vars=['invocation', 'acquisition', 'release'],  # Columns to stack
        var_name='event_type',                                # New column holding the old column names
        value_name='timestamp'                                # New column holding the actual timestamps
    )

    # 2. Encode event_type as a compact int8 code instead of the melted strings.
    # This keeps the timeline purely numeric so downstream consumers never
    # materialize an object-dtype array (see analysis/defs.py).
    timeline_df['event_type'] = timeline_df['event_type'].map(EVENT_CODES).astype(np.int8)

    # 3. Sort the entire dataframe chronologically by timestamp
    timeline_df = timeline_df.sort_values(by='timestamp', ignore_index=True)

    return timeline_df