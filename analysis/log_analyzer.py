
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Tuple, Optional

from .defs import INVOCATION, ACQUISITION

class LogAnalyzer:

    def __init__(self, data: pd.DataFrame, global_timeline: pd.DataFrame, overtake_timeline: Optional[pd.DataFrame]=None):
        self._data = data
        self._global_timeline = global_timeline
        self._overtake_timeline = overtake_timeline if overtake_timeline is not None else self.create_overtake_timeline()
        self.num_threads = data['thread_id'].values[-1] + 1  # Assuming thread IDs are 0-indexed and contiguous
        self.operation_count = len(data)
        self.event_count = len(global_timeline)

    def close(self) -> None:
        self._data = None
        self._global_timeline = None
        self._overtake_timeline = None

    def create_overtake_timeline(self) -> pd.DataFrame:
        """
        Creates an operation-wise timeline of overtakes.
        O(N*t) complexity using a chronological state-tracker.
        """
        timeline = self._global_timeline
        
        if timeline.empty:
            return pd.DataFrame(columns=['invocation_time', 'thread_id', 'intervening_acquisitions'])

        # 1. Drop out of Pandas into native NumPy arrays to eliminate loop overhead.
        # Extract each column separately so every array keeps its own dtype: a
        # single to_numpy() over mixed uint64/int8/int64 columns would coerce the
        # whole thing to float64 (losing TSC precision) or object (huge). Assumes
        # the timeline is already chronologically sorted by timestamp.
        ts = timeline['timestamp'].to_numpy()
        ev = timeline['event_type'].to_numpy()
        tid = timeline['thread_id'].to_numpy()

        # State trackers
        pending_invocations = {}  # thread_id -> (invocation_time, result_index)
        results = []              # Will store lists of: [invocation_time, thread_id, overtake_count]

        # 2. Step through the chronological timeline O(N)
        for timestamp, event_type, thread_id in zip(ts, ev, tid):

            if event_type == INVOCATION:
                # Save its future index in the results array so we can increment its penalty later
                idx = len(results)
                results.append([timestamp, thread_id, 0])
                pending_invocations[thread_id] = (timestamp, idx)

            elif event_type == ACQUISITION:
                # The thread acquired the lock, remove it from the pending pool
                if thread_id in pending_invocations:
                    acq_inv_time, _ = pending_invocations.pop(thread_id)
                    
                    # 3. Check who it overtook O(t)
                    # Iterate through the remaining pending threads (max size = t - 1)
                    for other_thread, (other_inv_time, other_idx) in pending_invocations.items():
                        
                        # If the acquiring thread invoked AFTER the pending thread,
                        # it means the pending thread was overtaken.
                        if acq_inv_time > other_inv_time:
                            results[other_idx][2] += 1
                            
        # 4. Rebuild the final dataframe in one shot
        final_ops = pd.DataFrame(results, columns=['invocation_time', 'thread_id', 'intervening_acquisitions'])
        
        return final_ops
    
    def print_overtake(self, file_path: Path) -> None:
        """
        Utility function to print out the overtake timeline to parquet.
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self._overtake_timeline.to_parquet(file_path)

#############################
#
#        AVG METRICS
#
#############################

    def calculate_all(self):
        pass

    def find_avg_per_thread_wait_time(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the average and variance of wait time for each thread as well as number of wait events of each thread.
        Returns: (averages_array, variances_array, counts_array)
        """
        if self._data.empty:
            return np.zeros(self.num_threads), np.zeros(self.num_threads), np.zeros(self.num_threads)

        # 1. Group and calculate both mean and variance simultaneously
        stats = self._data.groupby('thread_id')['wait_time'].agg(['mean', 'var', 'count'])
        
        # 2. Reindex to ensure all threads (0 to num_threads - 1) are present
        aligned = stats.reindex(range(self.num_threads), fill_value=0.0)
        
        # 3. Pandas variance (ddof=1) returns NaN if a thread has only 1 data point. Fill with 0.0.
        aligned['var'] = aligned['var'].fillna(0.0)
        
        return aligned['mean'].to_numpy(), aligned['var'].to_numpy(), aligned['count'].to_numpy()

    def percent_time_in_CS(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the average and variance of the percentage of time spent in the 
        critical section per lock acquisition event, per thread.
        Returns: (averages_array, variances_array)
        """
        if self._data.empty:
            return np.zeros(self.num_threads), np.zeros(self.num_threads)

        # 1. Calculate total time per individual event
        total_time = self._data['wait_time'] + self._data['hold_time']
        
        # 2. Compute individual event percentages safely
        event_pct = np.where(total_time > 0, (self._data['hold_time'] / total_time) * 100, 0.0)
        
        # 3. Create a temporary dataframe to group by thread_id
        temp_df = pd.DataFrame({
            'thread_id': self._data['thread_id'],
            'pct_in_cs': event_pct
        })
        
        # 4. Group and aggregate
        stats = temp_df.groupby('thread_id')['pct_in_cs'].agg(['mean', 'var'])
        aligned = stats.reindex(range(self.num_threads), fill_value=0.0)
        aligned['var'] = aligned['var'].fillna(0.0)
        
        return aligned['mean'].to_numpy(), aligned['var'].to_numpy()

    def overtake_percentage(self) -> Tuple[float, float]:
        """
        Computes the mean percentage and variance of operations that are overtaken.
        Returns: (mean_percentage, variance_of_percentage)
        """
        counts = self._overtake_timeline['intervening_acquisitions']

        if counts.empty:
            return 0.0, 0.0

        # Create a boolean/float series (1.0 for overtaken, 0.0 for not)
        is_overtaken = (counts > 0).astype(float)

        # Mean is the fraction of True values. Multiply by 100 for percentage.
        mean_pct = float(is_overtaken.mean() * 100)
        
        # Variance of the 0/1 occurrences scaled to percentage
        var_pct = float(is_overtaken.var() * (100 ** 2))
        
        if pd.isna(var_pct):
            var_pct = 0.0

        return mean_pct, var_pct
    
    def average_overtake_depth(self) -> Tuple[float, float]:
        """
        Computes the average and variance of the depth of elements with overtake depth > 0.
        Returns: (mean_depth, variance_depth)
        """
        overtake_depths = self._overtake_timeline['intervening_acquisitions']
        positive_depths = overtake_depths[overtake_depths > 0]

        if positive_depths.empty:
            return 0.0, 0.0

        mean_depth = float(positive_depths.mean())
        var_depth = float(positive_depths.var())
        
        if pd.isna(var_depth):
            var_depth = 0.0

        return mean_depth, var_depth
    

##############################
#
#       PER RUN METRICS
#
##############################

    def total_CS_completions(self) -> int:
        """
        Computes the total number of critical section completions
        """
        return int(len(self._data))

    def lock_transfer_matrix(self) -> np.ndarray:
        """
        Tracks which threads are most likely to aquire lock after some thread X releases lock
        Returns nxn np.array where n = thread count
        """

        n = self.num_threads
        transfer_table = np.zeros((n, n))
        
        # 1. Filter only acquisition events
        acquisitions = self._global_timeline[self._global_timeline['event_type'] == ACQUISITION]
        t_ids = acquisitions['thread_id'].to_numpy()
        
        # 2. 'from_threads' are elements 0 to N-1
        #    'to_threads'   are elements 1 to N
        # This aligns the previous owner with the next owner perfectly.
        from_threads = t_ids[:-1]
        to_threads = t_ids[1:]
        
        # 3. Use np.add.at to handle duplicate transfers correctly
        # transfer_table[from, to] += 1
        np.add.at(transfer_table, (from_threads, to_threads), 1)
        
        return transfer_table


    def mean_squared_rank_inversion_penalty(self) -> Tuple[float, float]:
        """
        Computes the root mean rank inversion penalty across all operations.
        k = 2, or quadratic penalty
        one operation overtaken 3 times is worse than 3 operations overtaken once each
        """
        if self._overtake_timeline.empty:
            return 0.0, 0.0

        # Vectorized squaring and sum using native pandas methods
        total_rip = self._overtake_timeline['intervening_acquisitions'] ** 2
        avg_rip = float(total_rip.mean())
        var_rip = float(total_rip.var())
        return avg_rip, var_rip
    