
import numpy as np
import pandas as pd

from typing import Tuple

class LogAnalyzer:

    def __init__(self, data: pd.DataFrame, global_timeline: pd.DataFrame):
        self._data = data
        self._global_timeline = global_timeline
        self._overtake_timeline = self.create_overtake_timeline()
        self.num_threads = data['thread_id'].values[-1] + 1  # Assuming thread IDs are 0-indexed and contiguous
        self.operation_count = len(data)
        self.event_count = len(global_timeline)

    def create_overtake_timeline(self) -> pd.DataFrame:
        """
        Creates an operation-wise timeline of overtakes.
        Operations are ordered based on invocation time.
        """
        timeline = self._global_timeline
        
        if timeline.empty:
            return pd.DataFrame(columns=['invocation_time', 'thread_id', 'intervening_acquisitions'])

        # 1. Create the cumulative sum of all acquisitions
        acq_cumsum = (timeline['event_type'] == 'acquisition').cumsum()
        
        # 2. Isolate invocations and acquisitions, tagging them with an operation sequence number per thread
        inv_mask = timeline['event_type'] == 'invocation'
        acq_mask = timeline['event_type'] == 'acquisition'
        
        inv_df = pd.DataFrame({
            'thread_id': timeline.loc[inv_mask, 'thread_id'],
            'invocation_time': timeline.loc[inv_mask, 'timestamp'],
            'counts_at_inv': acq_cumsum[inv_mask],
            'op_seq': timeline[inv_mask].groupby('thread_id').cumcount()
        })
        
        acq_df = pd.DataFrame({
            'thread_id': timeline.loc[acq_mask, 'thread_id'],
            'counts_at_acq': acq_cumsum[acq_mask],
            'op_seq': timeline[acq_mask].groupby('thread_id').cumcount()
        })
        
        # 3. Merge the two sets so each invocation lines up with its corresponding acquisition
        merged = pd.merge(inv_df, acq_df, on=['thread_id', 'op_seq'])
        
        # 4. Calculate intervening counts
        merged['intervening_acquisitions'] = merged['counts_at_acq'] - merged['counts_at_inv'] - 1
        
        # 5. Filter to the required columns and sort by invocation time
        final_ops = merged[['invocation_time', 'thread_id', 'intervening_acquisitions']]
        final_ops = final_ops.sort_values(by='invocation_time', ignore_index=True)
        
        return final_ops

#############################
#
#        AVG METRICS
#
#############################

    def calculate_all(self):
        pass

    def find_avg_per_thread_wait_time(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the average and variance of wait time for each thread.
        Returns: (averages_array, variances_array)
        """
        if self._data.empty:
            return np.zeros(self.num_threads), np.zeros(self.num_threads)

        # 1. Group and calculate both mean and variance simultaneously
        stats = self._data.groupby('thread_id')['wait_time'].agg(['mean', 'var'])
        
        # 2. Reindex to ensure all threads (0 to num_threads - 1) are present
        aligned = stats.reindex(range(self.num_threads), fill_value=0.0)
        
        # 3. Pandas variance (ddof=1) returns NaN if a thread has only 1 data point. Fill with 0.0.
        aligned['var'] = aligned['var'].fillna(0.0)
        
        return aligned['mean'].to_numpy(), aligned['var'].to_numpy()

    def percent_time_in_CS(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the average and variance of the percentage of time spent in the 
        critical section per lock acquisition event, per thread.
        Returns: (averages_array, variances_array)
        """
        if self._data.empty:
            return np.zeros(self.num_threads), np.zeros(self.num_threads)

        # 1. Calculate total time per individual event
        total_time = self._data['wait_times'] + self._data['hold_times']
        
        # 2. Compute individual event percentages safely
        event_pct = np.where(total_time > 0, (self._data['hold_times'] / total_time) * 100, 0.0)
        
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
        acquisitions = self._global_timeline[self._global_timeline['event_type'] == 'acquisition']
        t_ids = acquisitions['thread_id']
        
        # 2. 'from_threads' are elements 0 to N-1
        #    'to_threads'   are elements 1 to N
        # This aligns the previous owner with the next owner perfectly.
        from_threads = t_ids[:-1]
        to_threads = t_ids[1:]
        
        # 3. Use np.add.at to handle duplicate transfers correctly
        # transfer_table[from, to] += 1
        np.add.at(transfer_table, (from_threads, to_threads), 1)
        
        return transfer_table


    def root_mean_rank_inversion_penalty(self) -> Tuple[float, float]:
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
        rmrip = np.sqrt(avg_rip)
        return rmrip, var_rip
    